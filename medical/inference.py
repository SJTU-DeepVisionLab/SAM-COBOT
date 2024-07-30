# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
#from segment_anything_lora import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import argparse
import traceback
import svf_torch
import random
import torch.nn as nn 
from utils.SurfaceDice import compute_dice_coefficient
from quaternion_layers import *
import seaborn as sns
import matplotlib.pyplot as plt
import ipdb


def init_weights(m):
	if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
		nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
		if m.bias is not None:
			truncated_normal_(m.bias, mean=0, std=0.001)
	if type(m) == nn.Linear:
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	if type(m) == nn.BatchNorm2d:
		nn.init.uniform_(m.weight)
		nn.init.constant_(m.bias, 0)

def truncated_normal_(tensor, mean=0, std=1):
	size = tensor.shape
	tmp = tensor.new_empty(size + (4,)).normal_()
	valid = (tmp < 2) & (tmp > -2)
	ind = valid.max(-1, keepdim=True)[1]
	tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
	tensor.data.mul_(std).add_(mean)

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[..., : self.dim] += new_q
        qkv[..., -self.dim:] += new_v
        return qkv


class Adapter(nn.Module):
    def __init__(self,
                 mlp: nn.Module,
                 d_model=768,
                 bottleneck=16,
                 dropout=0.1,
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model
        self.mlp = mlp
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = self.mlp(x) if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output



class Adapter_relation(nn.Module):
    def __init__(
                self,
                mlp: nn.Module,
                FacT_c: nn.Parameter,
                FacT_d1:nn.Parameter,
                FacT_p: nn.Parameter,
                idx: int,
                d_model=768,
                bottleneck=16,
                dropout=0.1,
                adapter_scalar="1.0",
                adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model
        self.mlp = mlp
        self.down_size = bottleneck
        self.FacT_c = FacT_c
        self.FacT_d1 = FacT_d1
        self.idx = idx
        self.FacT_p = FacT_p
        layer_scale_init_value = 1.0 
        dim = 16
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = self.mlp(x) if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        self.CP = (self.FacT_c @ self.FacT_p)[..., self.idx]
        down = self.FacT_d1(self.down_proj(x).mul(self.CP) + self.gamma * self.down_proj(x))



        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output



class _LoRA_qkv_relation_add(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            FacT_c: nn.Parameter,
            FacT_d1:nn.Parameter,
            FacT_d2:nn.Parameter,
            FacT_p: nn.Parameter,
            idx: int,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.idx = idx
        self.FacT_c = FacT_c
        self.FacT_d1 = FacT_d1
        self.FacT_d2 = FacT_d2

        self.FacT_p = FacT_p
        self.w_identity = torch.eye(qkv.in_features)
        layer_scale_init_value = 1.0 
        dim = 8
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        self.CP = (self.FacT_c @ self.FacT_p)[..., self.idx:self.idx+2]



        new_q = self.linear_b_q(self.FacT_d1(self.linear_a_q(x).mul(self.CP[:,:,0]) + self.gamma[:4] * self.linear_a_q(x)))
        new_v = self.linear_b_v(self.FacT_d2(self.linear_a_v(x).mul(self.CP[:,:,1]) + self.gamma[-4:] * self.linear_a_v(x)))
        qkv[..., : self.dim] += new_q
        qkv[..., -self.dim:] += new_v
        return qkv


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum




#%% run inference
# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set based on MedSAM')
parser.add_argument('-i', '--data_path', type=str, default='data/Npz_files/CT_Abd-Gallbladder/test', help='path to the data folder')
parser.add_argument('-o', '--seg_path_root', type=str, default='data/Test_MedSAMBaseSeg', help='path to the segmentation folder')
parser.add_argument('--seg_png_path', type=str, default='data/sanity_test/Test_MedSAMBase_png', help='path to the segmentation folder')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--device', type=str, default='cuda:0', help='device')

# checkpoint type six  ps:TC_relation means LORA+relation
parser.add_argument('--checkpoint_LORA', type=str, default='./work_dir_LORA_50/CT_Abd-Gallbladder/sam_model_best.pth')
parser.add_argument('--checkpoint_LORA_relation', type=str, default='./work_dir_LORA_relation_lr*100_linear/CT_Abd-Gallbladder/sam_model_best.pth')
parser.add_argument('--checkpoint_base', type=str, default='./work_dir_TAM_baseline_50/CT_Abd-Gallbladder/sam_model_best.pth')
parser.add_argument('--checkpoint_adapter', type=str, default='./work_dir_TAM_Adapter_50/CT_Abd-Gallbladder/sam_model_best.pth')
parser.add_argument('--checkpoint_adapter_relation', type=str, default='./work_dir_TAM_Adapter_relation_lr*10/CT_Abd-Gallbladder/sam_model_best.pth')
parser.add_argument('--checkpoint_initial', type=str, default='./work_dir/SAM/sam_vit_b_01ec64.pth')

parser.add_argument('--adaptation_type', type=str, default='Adapter', help='self_PET_adaptation')

parser.add_argument('--seed', default=1234, type=int)
args = parser.parse_args()

device = args.device
seed = args.seed
# cudnn.benchmark = True
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

if args.adaptation_type == 'scratch':
    sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)
    sam_model_tune.eval()


elif args.adaptation_type == 'baseline':
    sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint_base).to(device)
    sam_model_tune.eval()


elif args.adaptation_type == 'LORA':

    sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)
    rank = 4
    for blk in sam_model_tune.image_encoder.blocks:
        w_qkv_linear = blk.attn.qkv
        dim = w_qkv_linear.in_features
        w_a_linear_q = nn.Linear(dim, rank, bias=False)
        w_b_linear_q = nn.Linear(rank, dim, bias=False)
        w_a_linear_v = nn.Linear(dim, rank, bias=False)
        w_b_linear_v = nn.Linear(rank, dim, bias=False)

        w_a_linear_q.apply(init_weights)
        w_b_linear_q.apply(init_weights)
        w_a_linear_v.apply(init_weights)
        w_b_linear_v.apply(init_weights)

        blk.attn.qkv = _LoRA_qkv(
            w_qkv_linear,
            w_a_linear_q,
            w_b_linear_q,
            w_a_linear_v,
            w_b_linear_v,
        )   


    sam_model_tune.load_state_dict(torch.load(args.checkpoint_LORA))
    

elif args.adaptation_type == 'LORA_relation':
    sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)
    rank = 4
    t_len = 24
    attention_len = 2
    mlp_len = 4
    idx = 0
    symbol = 1

    FacT_c = nn.Parameter(torch.empty(1, rank, t_len))
    FacT_p = nn.Parameter(torch.empty(t_len, t_len))
    nn.init.ones_(FacT_c) 

    nn.init.eye_(FacT_p)
    w_qkv_linear_dim = sam_model_tune.image_encoder.blocks[0].attn.qkv
    dim = w_qkv_linear_dim.in_features




    for blk in sam_model_tune.image_encoder.blocks:
        w_a_linear_q = nn.Linear(dim, rank, bias=False)
        w_b_linear_q = nn.Linear(rank, dim, bias=False)
        w_a_linear_v = nn.Linear(dim, rank, bias=False)
        w_b_linear_v = nn.Linear(rank, dim, bias=False)
        FacT_d1 = QuaternionLinear(rank, rank, bias=False)
        FacT_d2 = QuaternionLinear(rank, rank, bias=False)
        w_a_linear_q.apply(init_weights)
        w_a_linear_v.apply(init_weights)
        w_b_linear_q.apply(init_weights)
        w_b_linear_v.apply(init_weights)
        
        w_qkv_linear = blk.attn.qkv
        blk.attn.qkv = _LoRA_qkv_relation_add(
            w_qkv_linear,
            w_a_linear_q,
            w_b_linear_q,
            w_a_linear_v,
            w_b_linear_v,
            FacT_c,
            FacT_d1,
            FacT_d2,
            FacT_p,
            idx,
        )
        idx += attention_len
    sam_model_tune.load_state_dict(torch.load(args.checkpoint_LORA_relation))

elif args.adaptation_type == 'Adapter':
    sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)

    for blk in sam_model_tune.image_encoder.blocks:

        blk.mlp = Adapter(blk.mlp)
            #print_trainable_parameters(self.model) 
    sam_model_tune.load_state_dict(torch.load(args.checkpoint_adapter))
    #validate_gram(sam_model_tune)
elif args.adaptation_type == 'Adapter_relation':

    sam_model_tune = sam_model_registry[args.model_type](checkpoint=args.checkpoint_initial).to(device)

    rank = 16
    t_len = 12   
    idx = 0   
    FacT_c = nn.Parameter(torch.empty(1, rank, t_len))
    FacT_p = nn.Parameter(torch.empty(t_len, t_len))

    nn.init.ones_(FacT_c) 
    nn.init.eye_(FacT_p)   

         
    for blk in sam_model_tune.image_encoder.blocks:
        FacT_d1 = QuaternionLinear(rank, rank, bias=False)
        blk.mlp = Adapter_relation(blk.mlp,
                                    FacT_c,
                                    FacT_d1,
                                    FacT_p,
                                    idx,
                                    )
        idx += 1
    sam_model_tune.load_state_dict(torch.load(args.checkpoint_adapter_relation))

sam_model_tune = sam_model_tune.to(device)
sam_model_tune.eval()

sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)

test_npzs = sorted(os.listdir(args.data_path))

imgs = []
gts = []


for npz_idx in range(0, len(test_npzs)):
    npz = np.load(join(args.data_path, test_npzs[npz_idx]))
    imgx = npz['imgs']
    gtx = npz['gts']
    imgs.append(imgx)
    gts.append(gtx)

imgs = np.concatenate(imgs)
gts = np.concatenate(gts)

def get_bbox_from_mask(mask):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return np.array([x_min, y_min, x_max, y_max])

ori_sam_segs = []
medsam_segs = []
medsam_segs_l = []
bboxes = []
for img, gt in zip(imgs, gts):
    bbox_initial = get_bbox_from_mask(gt)
    bboxes.append(bbox_initial)
    # predict the segmentation mask using the fine-tuned model
    H, W = img.shape[:2]
    resize_img = sam_trans.apply_image(img)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model_tune.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    #input_image2 = sam_model_tune_baseline.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        bbox = sam_trans.apply_boxes(bbox_initial, (H, W))
        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        medsam_segs.append(medsam_seg)



#%% compute the DSC score
medsam_segs = np.stack(medsam_segs, axis=0)
medsam_dsc = compute_dice_coefficient(gts>0, medsam_segs>0)
print('SAM_COBOT DSC: {:.4f}'.format(medsam_dsc))
