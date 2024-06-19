#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.segment_anything.modeling.common import LayerNorm2d
from model.segment_anything.modeling.image_encoder import Block
from model.segment_anything import sam_model_registry
from model.mobile_encoder.setup_mobile_sam import setup_model as build_sam_mobile
from model.quaternion_layers import *
from model.loss_functions import dice_loss, multilabel_dice_loss
from model.utils import init_weights
import ipdb
import svf_torch
import loralib as lora
import math

from typing import Optional, List

# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for name, param in model.named_parameters():
#         if 'image_encoder' in name:
#             all_param += param.numel()
#             if param.requires_grad:
#                 print(name)
#                 trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.6f}"
#     )




class Adapter(nn.Module):
    def __init__(self,
                 mlp: nn.Module,
                 d_model=160,
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

    


class SonarSAM(nn.Module):
    def __init__(self, model_name, checkpoint, num_classes=12, 
                 is_finetune_image_encoder=False,
                 use_adaptation=False,
                 adaptation_type='learnable_prompt_layer',
                 head_type='custom',
                 reduction=4, upsample_times=2, groups=4, rank=4) -> None:
        super(SonarSAM, self).__init__()
        
        #load same from the pretrained model
        if model_name == 'mobile':
            self.sam = build_sam_mobile(checkpoint=checkpoint)
        else:
            self.sam = sam_model_registry[model_name](checkpoint=checkpoint, num_multimask_outputs=num_classes)
        self.is_finetune_image_encoder = is_finetune_image_encoder
        self.use_adaptation = use_adaptation
        self.adaptation_type = adaptation_type
        self.head_type = head_type
        self.num_classes = num_classes
        self.model_name = model_name

        # freeze image encoder
        if not self.is_finetune_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        if self.use_adaptation:     
            if self.adaptation_type == 'LORA':
                if self.model_name != 'mobile':
                    for blk in self.sam.image_encoder.blocks:
                        w_qkv_linear = blk.attn.qkv
                        self.dim = w_qkv_linear.in_features
                        w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
                        w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
                        w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
                        w_b_linear_v = nn.Linear(rank, self.dim, bias=False)

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
                else:
                    for i_layer in range(1, len(self.sam.image_encoder.layers)):
                        for blk in self.sam.image_encoder.layers[i_layer].blocks:
                            w_qkv_linear = blk.attn.qkv
                            self.dim = w_qkv_linear.in_features
                            w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
                            w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
                            w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
                            w_b_linear_v = nn.Linear(rank, self.dim, bias=False)

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

            elif self.adaptation_type == 'Adapter':
                #for i_layer in range(1, len(self.sam.image_encoder.layers)):
                for blk in self.sam.image_encoder.blocks:
                    w_qkv_linear = blk.attn.qkv
                    dim = w_qkv_linear.in_features
                    # w_qkv_linear = blk.attn.qkv
                    # self.dim = w_qkv_linear.in_features
            #for blk in self.sam.image_encoder.blocks:

                    blk.mlp = Adapter(blk.mlp, dim)


            elif self.adaptation_type == 'Adapter_relation':
                rank = 16
                t_len = 12   
                idx = 0   
                FacT_c = nn.Parameter(torch.empty(1, rank, t_len))
                FacT_p = nn.Parameter(torch.empty(t_len, t_len))

                nn.init.ones_(FacT_c) 
                nn.init.eye_(FacT_p)            
                for blk in self.sam.image_encoder.blocks:
                    w_qkv_linear = blk.attn.qkv
                    dim = w_qkv_linear.in_features
                    FacT_d1 = QuaternionLinear(rank, rank, bias=False)
                    blk.mlp = Adapter_relation(blk.mlp,
                                            FacT_c,
                                            FacT_d1,
                                            FacT_p,
                                            idx,
                                            )
                    idx += 1

            elif self.adaptation_type == 'TAM_CP_relation':
                rank = 4
                t_len = 24
                #t_len2 = 48
                attention_len = 2
                mlp_len = 4
                idx = 0
                symbol = 1
                FacT_c = nn.Parameter(torch.empty(1, rank, t_len))

                FacT_p = nn.Parameter(torch.empty(t_len, t_len))

                nn.init.ones_(FacT_c) 

                nn.init.eye_(FacT_p)
                for blk in self.sam.image_encoder.blocks:
                    w_qkv_linear = blk.attn.qkv
                    self.dim = w_qkv_linear.in_features
                    w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
                    w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
                    w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
                    w_b_linear_v = nn.Linear(rank, self.dim, bias=False)
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

            #print_trainable_parameters(self.model)

        if 'SVD' in self.adaptation_type:
            out_dim = self.sam.image_encoder.neck[0].conv_U.out_channels
        elif 'NECK_LORA' in self.adaptation_type:
            out_dim = self.sam.image_encoder.neck[0].conv.out_channels
        else:
            out_dim = self.sam.image_encoder.neck[0].out_channels
            
        self.img_size = self.sam.image_encoder.img_size


    def upscale(self, x, times=2):
        for i in range(times):
            x = self.up_conv["up_{}".format(i+1)](x)
        return x

    def forward(self, x, boxes=None):
        out = self.sam.image_encoder(x)
        seg_out = []
        if self.head_type in ['semantic_mask_decoder', 'semantic_mask_decoder_LORA']:            
            for idx, curr_embedding in enumerate(out):
                points = None
                if boxes is not None:
                    bboxes = boxes[idx]                    
                else:
                    bboxes = None
                masks = None
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=points,
                    boxes=bboxes,
                    masks=masks,
                )
                
                low_res_masks, iou_predictions = self.sam.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # print('low res masks', low_res_masks.shape)
                masks = F.interpolate(low_res_masks, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
                # print('masks', masks.shape)
                # low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
                output = []
                for idx in range(masks.shape[0]):                    
                    mask = masks[idx, ...]
                    # print('mask', mask.shape)                                        
                    output.append(mask.squeeze())
                seg_out.append(output)
        else:
            raise ValueError('unknow head type: {}'.format(self.head_type))

        return seg_out


class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.bcewithlogit = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = dice_loss

    def forward(self, images, masks, boxes):
        pred_masks = self.model(images, boxes)
        bce_loss, dice = 0, 0
        for idx, im_masks in enumerate(masks):
            assert len(pred_masks) == len(masks)
            p_im_masks = pred_masks[idx]
            for p_m, m in zip(p_im_masks, im_masks):                
                bce_loss += self.bcewithlogit(input=p_m, target=m)
                dice += self.dice_loss(label=m.unsqueeze(0), mask=p_m.unsqueeze(0))
        loss = bce_loss + dice
        return loss, pred_masks
    

if __name__ == "__main__":
    with torch.no_grad():                 
        model = SonarSAM("vit_b", "ckpts/sam_vit_b_01ec64.pth", 
                         num_classes=12, 
                         is_finetune_image_encoder=False, 
                         use_adaptation=False, 
                         adaptation_type='LORA', 
                         head_type='semantic_mask_decoder',
                         reduction=4, upsample_times=2, groups=4, rank=4).half().cuda()
        x = torch.randn(1, 3, 1024, 1024).half().cuda()

        out = model(x)
        print(out.shape)
