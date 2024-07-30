# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn as nn
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
from quaternion_layers import *
# set seeds
from torch.cuda.amp import autocast
torch.manual_seed(2023)
np.random.seed(2023)
import ipdb
import svf_torch
#import torchvision.models as models

#model = models.resnet18(pretrained=True)

def count_1x1_convolutions(model):
    count = 0
    count_layers = []
    count_layer = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (1, 1):
            count += 1
            count_layers.append(count_layer)
        count_layer += 1
    return count, count_layers

def count_3x3_convolutions(model):
    count = 0
    count_layers = []
    count_layer = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (3, 3):
            count += 1
            count_layers.append(count_layer)
        count_layer += 1
    return count, count_layers

def transform(img):
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size) 
    resize_img = sam_transform.apply_image(img)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    # model input: (1, 3, 1024, 1024)
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    return input_image

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if 'image_encoder' in name:
            all_param += param.numel()
            if param.requires_grad:
                print(name)
                trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {50 * trainable_params / all_param:.6f}"
    )

#%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()


#%% create a dataset class to load npz data and return back image and ground truth
class NpzimgDataset(Dataset): 
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        self.imgs = np.vstack([d['imgs'] for d in self.npz_data])
        self.transform = transform
        #print(f"{self.img_embeddings.shape=}, {self.ori_gts.shape=}")
    
    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        input_image = self.imgs[index]
        input_image = self.transform(input_image).squeeze(0)
        #resize_img = sam_transform.apply_image(img)
        #resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to('cuda:0')
    # model input: (1, 3, 1024, 1024)
        #input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)

        gt2D = self.ori_gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return input_image, torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float()


# %% test dataset class and dataloader
npz_tr_path = 'data/Npz_files/CT_Abd-Gallbladder/train'
demo_dataset = NpzDataset(npz_tr_path)
demo_dataloader = DataLoader(demo_dataset, batch_size=8, shuffle=True)
for img_embed, gt2D, bboxes in demo_dataloader:
    # img_embed: (B, 256, 64, 64), gt2D: (B, 1, 256, 256), bboxes: (B, 4)
    print(f"{img_embed.shape=}, {gt2D.shape=}, {bboxes.shape=}")
    break

# %% set up model for fine-tuning 
# train data path
npz_tr_path = 'data/Npz_files/CT_Abd-Gallbladder/train'
work_dir = './work_dir_CVPR2024_rebuttal_ablation_only_HL'
task_name = 'CT_Abd-Gallbladder'
# prepare SAM model
model_type = 'vit_b'
checkpoint = 'work_dir/SAM/sam_vit_b_01ec64.pth'
device = 'cuda:3'
self_adapation_type = 'Adapter_relation_ablation_only_HL' # TAM_CP_relatio
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)#.double()

for param in sam_model.image_encoder.parameters():
    param.requires_grad = False
    
for param in sam_model.prompt_encoder.parameters():
    param.requires_grad = False

for param in sam_model.mask_decoder.parameters():
        param.requires_grad = True


#for name, param in sam_model.named_parameters():
#    print(name)
#ipdb.set_trace()    
def init_weights(m):
	if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
		nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
		# nn.init.normal_(m.weight, std=0.001)
		# nn.init.normal_(m.bias, std=0.001)
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


class Adapter(nn.Module):
    def __init__(self,
                 mlp: nn.Module,
                 d_model=768,
                 bottleneck=8,
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
        layer_scale_init_value = 1.0 #0#torch.#math.exp(0) + 1e-8   # 1.0

        #self.epsilon = torch.tensor(epsilon, dtype=torch.float)
        dim = 16
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
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
        self.CP = (self.FacT_c @ self.FacT_p)[..., self.idx]

        down = self.FacT_d1(self.down_proj(x).mul(self.CP) + self.gamma * self.down_proj(x))
        # ablation1
        #down = self.down_proj(x).mul(self.CP) + self.gamma * self.down_proj(x)
        # ablation2
        #down = self.FacT_d1(self.gamma * self.down_proj(x))

        # ablation3
        #down = self.gamma * self.down_proj(x)

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
        # qkv[:, :, :, : self.dim] += new_q
        # qkv[:, :, :, -self.dim:] += new_v
        qkv[..., : self.dim] += new_q
        qkv[..., -self.dim:] += new_v
        return qkv

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
            #nu: float = 1.0,
            #epsilon: float = 1e-8,
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
        layer_scale_init_value = 1.0 #0#torch.#math.exp(0) + 1e-8   # 1.0

        #self.epsilon = torch.tensor(epsilon, dtype=torch.float)
        dim = 8
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        # self.gamma2 = torch.exp(nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                           requires_grad=True)) + self.epsilon 

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        #self.CP = ((self.FacT_c @ self.FacT_d)@ self.FacT_p)[..., self.idx:self.idx+2]
        #FacT_c1 = torch.stack([torch.diag(self.FacT_c[:, i]) for i in range(self.FacT_c.size(1))], dim=2)
        self.CP = (self.FacT_c @ self.FacT_p)[..., self.idx:self.idx+2]



        new_q = self.linear_b_q(self.FacT_d1(self.linear_a_q(x).mul(self.CP[:,:,0]) + self.gamma[:4] * self.linear_a_q(x)))
        new_v = self.linear_b_v(self.FacT_d2(self.linear_a_v(x).mul(self.CP[:,:,1]) + self.gamma[-4:] * self.linear_a_v(x)))
        # qkv[:, :, :, : self.dim] += new_q
        # qkv[:, :, :, -self.dim:] += new_v
        qkv[..., : self.dim] += new_q
        qkv[..., -self.dim:] += new_v
        return qkv

if self_adapation_type == 'baseline':

    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
        
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False

    for param in sam_model.mask_decoder.parameters():
         param.requires_grad = True


elif self_adapation_type == 'LORA':

    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
        
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False

    for param in sam_model.mask_decoder.parameters():
         param.requires_grad = True
    rank = 16
    #ipdb.set_trace()
    #for i_layer in range(1, len(self.model.image_encoder.layers)):
        #for blk in self.model.image_encoder.layers[i_layer].blocks:
    for blk in sam_model.image_encoder.blocks:
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


elif self_adapation_type == 'LORA_relation':

    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
        
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False

    for param in sam_model.mask_decoder.parameters():
         param.requires_grad = True

    rank = 4
    t_len = 24
    #t_len2 = 48
    attention_len = 2
    mlp_len = 4
    idx = 0
    symbol = 1

    FacT_c = nn.Parameter(torch.empty(1, rank, t_len))
    #FacT_d = nn.Parameter(torch.empty(t_len, t_len2))
    FacT_p = nn.Parameter(torch.empty(t_len, t_len))

    nn.init.ones_(FacT_c) 

    nn.init.eye_(FacT_p)
    w_qkv_linear_dim = sam_model.image_encoder.blocks[0].attn.qkv
    dim = w_qkv_linear_dim.in_features




    for blk in sam_model.image_encoder.blocks:
        w_a_linear_q = nn.Linear(dim, rank, bias=False)
        w_b_linear_q = nn.Linear(rank, dim, bias=False)
        w_a_linear_v = nn.Linear(dim, rank, bias=False)
        w_b_linear_v = nn.Linear(rank, dim, bias=False)
        FacT_d1 = nn.Linear(rank, rank, bias=False)
        FacT_d2 = nn.Linear(rank, rank, bias=False)
        #FacT_d1 = QuaternionLinear(rank, rank, bias=False)
        #FacT_d2 = QuaternionLinear(rank, rank, bias=False)
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


elif self_adapation_type == 'Adapter':
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
        
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False

    for param in sam_model.mask_decoder.parameters():
         param.requires_grad = True

    for blk in sam_model.image_encoder.blocks:

        blk.mlp = Adapter(blk.mlp)
            #print_trainable_parameters(self.model) 

elif self_adapation_type == 'Adapter_relation':

    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
        
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False

    for param in sam_model.mask_decoder.parameters():
         param.requires_grad = True

    rank = 16
    t_len = 12   
    idx = 0   
    FacT_c = nn.Parameter(torch.empty(1, rank, t_len), requires_grad=True)
    FacT_p = nn.Parameter(torch.empty(t_len, t_len), requires_grad=True)

    nn.init.ones_(FacT_c) 
    nn.init.eye_(FacT_p)   

         
    for blk in sam_model.image_encoder.blocks:
        FacT_d1 = QuaternionLinear(rank, rank, bias=False)
        blk.mlp = Adapter_relation(blk.mlp,
                                    FacT_c,
                                    FacT_d1,
                                    FacT_p,
                                    idx,
                                    )
        idx += 1
    #print_trainable_parameters(self.model) 



#ipdb.set_trace()
sam_model = sam_model.to(device)
sam_model.train()


other_params = [p for n, p in sam_model.named_parameters() if "FacT_c" not in n and "FacT_p" not in n and "FacT_d" not in n]
qu_params = [p for n, p in sam_model.named_parameters() if "FacT_c" in n or "FacT_p" in n or "FacT_d" in n]
#other_params = [p for n, p in model.model.named_parameters() if "pos_bias_p" not in n]
#qu_params = [p for n, p in model.model.named_parameters() if "pos_bias_p" in n]

params = [{'params': qu_params, 'lr': 1.25e-4},
        {'params': other_params, 'lr': 1.25e-6}]


optimizer = torch.optim.Adam(params, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
print_trainable_parameters(sam_model)

#%% train
num_epochs = 25
losses = []
best_loss = 1e10
train_dataset = NpzimgDataset(npz_tr_path, transform)
#ipdb.set_trace()
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
params = sam_model.state_dict()



for epoch in range(num_epochs):
    epoch_loss = 0
    # train
    for step, (image, gt2D, boxes) in enumerate(tqdm(train_dataloader)):

        image_embedding = sam_model.image_encoder(image.to(device))
        with torch.no_grad():
                # convert box to 1024x1024 grid
            box_np = boxes.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)
                # get prompt embeddings 
                #with autocast():
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )

        loss = seg_loss(mask_predictions, gt2D.to(device))
        #ipdb.set_trace()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= step
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    # save the latest model checkpoint
    torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_latest.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))

    # plot loss
    plt.plot(losses)
    plt.title('Dice + Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() # comment this line if you are running on a server
    plt.savefig(join(model_save_path, 'train_loss.png'))
    plt.close()

#%% compare the segmentation results between the original SAM model and the fine-tuned model
# load the original SAM model
ori_sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
ori_sam_predictor = SamPredictor(ori_sam_model)
npz_ts_path = 'data/Npz_files/CT_Abd-Gallbladder/test'
test_npzs = sorted(os.listdir(npz_ts_path))
# random select a test case
imgs = []
gts = []

#npz_idx = np.random.randint(0, len(test_npzs))
for npz_idx in range(0, len(test_npzs)):
    npz = np.load(join(npz_ts_path, test_npzs[npz_idx]))
    img = npz['imgs']
    gt = npz['gts']
    imgs.append(img)
    gts.append(gt)

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
bboxes = []
for img, gt in zip(imgs, gts):
    bbox = get_bbox_from_mask(gt)
    bboxes.append(bbox)
    # predict the segmentation mask using the original SAM model
    ori_sam_predictor.set_image(img)
    ori_sam_seg, _, _ = ori_sam_predictor.predict(point_coords=None, box=bbox, multimask_output=False)
    ori_sam_segs.append(ori_sam_seg[0])
    
    # predict the segmentation mask using the fine-tuned model
    H, W = img.shape[:2]
    resize_img = sam_trans.apply_image(img)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        bbox = sam_trans.apply_boxes(bbox, (H, W))
        box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
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
ori_sam_segs = np.stack(ori_sam_segs, axis=0)
medsam_segs = np.stack(medsam_segs, axis=0)
ori_sam_dsc = compute_dice_coefficient(gts>0, ori_sam_segs>0)
medsam_dsc = compute_dice_coefficient(gts>0, medsam_segs>0)
print('Original SAM DSC: {:.4f}'.format(ori_sam_dsc), 'MedSAM DSC: {:.4f}'.format(medsam_dsc))


#%% visualize the segmentation results of the middle slice
# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))    


img_id = int(imgs.shape[0]/2)  # np.random.randint(imgs.shape[0])
_, axs = plt.subplots(1, 3, figsize=(25, 25))
axs[0].imshow(imgs[img_id])
show_mask(gts[img_id], axs[0])
# show_box(box_np[img_id], axs[0])
# axs[0].set_title('Mask with Tuned Model', fontsize=20)
axs[0].axis('off')

axs[1].imshow(imgs[img_id])
show_mask(ori_sam_segs[img_id], axs[1])
show_box(bboxes[img_id], axs[1])
# add text to image to show dice score
axs[1].text(0.5, 0.5, 'SAM DSC: {:.4f}'.format(ori_sam_dsc), fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')
# axs[1].set_title('Mask with Untuned Model', fontsize=20)
axs[1].axis('off')

axs[2].imshow(imgs[img_id])
show_mask(medsam_segs[img_id], axs[2])
show_box(bboxes[img_id], axs[2])
# add text to image to show dice score
axs[2].text(0.5, 0.5, 'MedSAM DSC: {:.4f}'.format(medsam_dsc), fontsize=30, horizontalalignment='left', verticalalignment='top', color='yellow')
# axs[2].set_title('Ground Truth', fontsize=20)
axs[2].axis('off')
plt.show()  
plt.subplots_adjust(wspace=0.01, hspace=0)
# save plot
# plt.savefig(join(model_save_path, test_npzs[npz_idx].split('.npz')[0] + str(img_id).zfill(3) + '.png'), bbox_inches='tight', dpi=300)
plt.close()
