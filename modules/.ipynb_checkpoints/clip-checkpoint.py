"""
Implementation of CLIP model
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""

import os
import urllib
import hashlib
import warnings
import math
import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from typing import Tuple, Union
from .utils import log_info
from .cluster import get_deep_cluster, get_cluster_inter
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PromptResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None, block_id=1, args=None):
        """
        Args:
            block_id: the id the the block in the whole model, start from 1
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self,x:torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PromptTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None, args=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[PromptResidualAttentionBlock(width, heads, attn_mask, i + 1, args)
                                            for i in range(layers)])
    def forward(self, x: torch.Tensor):
        for i in range(self.layers):
            x = self.resblocks[i](x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None, block_id=1, args=None):
        """
        Args:
            block_id: the id the the block in the whole model, start from 1
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.visual_prompt_length = 4
        ############################################ NEW ADDED CODE ############################################
        '''
        self.block_id = block_id
        self.tokencluster_inter = get_cluster_inter(d_model, block_id, args)
        self.deepcluster = get_deep_cluster(d_model, block_id, args)
        '''

    def attention(self, q: torch.Tensor,k: torch.Tensor, v: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(q.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=q.dtype, device=q.device) if attn_mask_ is not None else None
        output = self.attn(q, k, v, need_weights=False, attn_mask=attn_mask_)[0]
        return output

    def forward(self, x_tuple:tuple):
        #x, video_frame, cluster_loss = x_tuple
        x, video_frame,visual = x_tuple

        if  visual:
            B = x.size(1)
            BT = B*video_frame
            T = video_frame
            dim = x.size(-1)
            visual_prompt,frame_token= x[:self.visual_prompt_length,:,:],x[self.visual_prompt_length:,:,:].reshape(-1,BT,dim)
            frame_token = self.ln_1(frame_token)
            visual_prompt = self.ln_1(visual_prompt)
            #attention1 attn_output_frames
            
            query1 = frame_token #  Frame tokens: [4+50, batch_size*num_frames, dim]
            
            key1 = torch.zeros(self.visual_prompt_length+query1.size(0),BT,dim).to(x.device)  #[4+49, batch_size*num_frames,dim]
            for i in range(0,BT,B):
                key1[:,i:i+B, :] = torch.cat((
                            visual_prompt,
                            query1[:, i:i+B, :]), dim=0)

            attention_output_frames = self.attention(query1,key1,key1).reshape(-1,B,dim) # [54*num_frames,batch_size, dim]

            #attention2 attn_output_proxies
            
            query2 = visual_prompt  # [4, batch_size, dim]
            key2 = torch.cat((visual_prompt,frame_token.reshape(-1,B,dim)),dim=0).to(x.device)   # [4+50*num_frames,batch_size,dim]

            attention_output_prompt = self.attention(query2,key2,key2)
            x = x + torch.cat((attention_output_prompt,attention_output_frames),dim=0) #  cancatenate: torch.cat([attn_output_proxies, attn_output_frames]
            #x = x + attention_output_frames

        else:
            x_ln = self.ln_1(x)
            x = x + self.attention(x_ln,x_ln,x_ln)
        # place 2, after self-attention
        x = x + self.mlp(self.ln_2(x))
        #return (x, video_frame, cluster_loss)
        return (x, video_frame,visual)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None, args=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, i + 1,args)
                                            for i in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1, visual=False):
        #cluster_loss = torch.zeros([], device=x.device)
        '''
        if not visual:
            return self.resblocks((x, video_frame, cluster_loss))[0]
        else:
            return self.resblocks((x, video_frame, cluster_loss))
        '''
        if not visual:
            return self.resblocks((x,video_frame,False))[0]
        else:
            return self.resblocks((x,video_frame,True))[0]




class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                     linear_patch: str = '2d',
                    video_frames=None, args=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.width = width
        assert linear_patch in ['2d', '3d']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2+1 , width))
        
        if args.time_embedding != 0: 
            self.frame_embedding = nn.Parameter(scale * torch.randn(video_frames,width).unsqueeze(1))
        else:
            self.frame_embedding = None
            
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, args=args)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        ############################################ NEW ADDED CODE ############################################
        self.linear_patch = linear_patch
        self.video_frames = video_frames
        # For 3D patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)
        # position ids (1, len_position_emb)
        self.register_buffer("position_ids", torch.arange(self.positional_embedding.shape[0]).expand(1, -1))
        self.num_tokens = 4

        

        #global prompt
        self.prompt_dropout =  nn.Dropout(0.0)
        self.prompt_proj = nn.Identity()
        prompt_dim = 768
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_tokens, prompt_dim))
        # xavier_uniform initialization
        patch_size = _pair(patch_size)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        ####code for visual prompt 
        '''
        patch_size = _pair(patch_size)
        num_tokens = 6
        self.num_tokens = num_tokens
        self.prompt_dropout =  nn.Dropout(0.0)
        prompt_dim = 768
        self.prompt_proj = nn.Identity()

        # random initiate prompt 
        
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
        with torch.no_grad():
            self.prompt_embeddings[:,0,:].data = self.class_embedding[None,None,:].data
        self.prompt_embeddings.requires_grad = True
        
        total_d_layer = layers-1
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
            total_d_layer, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        '''
        
    def incorporate_prompt(self, x, unified_visual_prompt):
        # combine prompt embeddings with image-patch embeddings

        BT = x.shape[0]
        B = BT//self.video_frames
        # after CLS token, all before image patches
        #x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)

        ## divide prompt
        unified_visual_frame_prompt = unified_visual_prompt.reshape(B,self.video_frames,self.num_tokens,x.size(-1))
       
        x = x.view(B,self.video_frames,x.size(-2),x.size(-1))
        

        unified_visual_global_prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))

        x_local_prompt = torch.cat((x[:,:,0:1,:],
                          unified_visual_frame_prompt,
                          x[:,:,1:,:],),dim=2).permute(0,2,1,3).reshape(B,-1,x.size(-1))

        x_prompt = torch.cat((unified_visual_global_prompt,x_local_prompt),dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x_prompt

    def forward_deep_prompt(self, x,unified_visual_prompt):

        ## x.shape L,N,D (N=BxT)
        attn_weights = []
        hidden_states = None
        weights = None
        B = x.shape[1]

        num_layers = self.transformer.layers

        for i in range(num_layers):
            if i == 0:
                ##(cls_token + n_prompt + n_patches,batch_size, hidden_dim) (55,768,768)
                hidden_states = self.transformer.resblocks[i]((x,self.video_frames,True))[0]
            else:
                if i <= len(unified_visual_prompt):
            
                    unified_visual_frame_prompt = unified_visual_prompt[i].reshape(B,self.video_frames,self.num_tokens,x.size(-1)).permute(2,1,0,3)
                    
                    hidden_states_global = hidden_states[:self.num_tokens, :, :]

                    hidden_states = hidden_states[self.num_tokens:, :, :].reshape(-1,self.video_frames,B,x.size(-1))
                    #hidden_states = hidden_states.reshape(-1,self.video_frames,B,x.size(-1))
                    hidden_states_local = torch.cat((
                        hidden_states[:1,:,:,:],
                        unified_visual_frame_prompt,
                        hidden_states[1+self.num_tokens:,:,:,:],
                    ), dim=0).reshape(-1,B,x.size(-1))

                    hidden_states = torch.cat((hidden_states_global,hidden_states_local),dim=0)

                hidden_states = self.transformer.resblocks[i]((hidden_states,self.video_frames,True))[0]

        #    if self.transformer.vis:
        #        attn_weights.append(weights)
        #encoded = self.encoder.encoder_norm(hidden_states)
        return hidden_states

    def forward(self, x: torch.Tensor,unified_visual_prompt, video_frame=-1):
        if x.ndim == 5: B, T, C, H, W = x.shape
        if x.ndim == 4:
            BT, C, H, W = x.shape
            B = BT // video_frame

        if self.linear_patch == '3d':
            assert video_frame != -1
            # [B, T, C, H, W]
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            # [B, C, T, H, W]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            # [B, width, T, grid, grid], grid = H // patch_size
            x_3d = self.conv2(x_3d)		
            # [B, T, width, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            # shape = [B x T, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() 
        else:
            # [B x T, width, grid, grid]
            x = self.conv1(x)
        # [B x T, width, grid x grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # [B x T, grid x grid, width]
        x = x.permute(0, 2, 1)

        # shape = [B x T, grid x grid + 1, width]
        '''
        if self.frame_embedding is not None:
            frame_embedding = self.frame_embedding.repeat(B,1,1).reshape(B,video_frame,1,self.width)
            #print('frame_embedding',frame_embedding.reshape(BT // video_frame, -1, self.width).shape)
            x = (x.reshape(B, video_frame, -1, self.width) + frame_embedding.to(x.dtype)).reshape(BT, -1, self.width)
        '''
        x = torch.cat([self.class_embedding.to(x.dtype) + \
                        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        
        #x = x + self.positional_embedding[1:,:].to(x.dtype)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        
        ### add for visual prompt
        ## shallow
        #self.prompt_embeddings = torch.cat((self.class_embedding[None,None,:],self.prompt_embeddings_),dim=1)
        #visual_prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)) #(B,4,dim)

        x = self.incorporate_prompt(x,unified_visual_prompt[0])

        x = x.permute(1, 0, 2)  					# NLD -> LND

        #x = x.reshape(-1,B,video_frame,x.size(-1)).permute(0,2,1,3).reshape(-1,B,x.size(-1))
        #x = torch.cat((visual_prompt.permute(1,0,2),x),dim=0)

        # org forward 
        #x = self.transformer(x, video_frame=video_frame, visual=True)
        x= self.forward_deep_prompt(x,unified_visual_prompt)
        #x = x[self.num_tokens:,:,:].reshape(-1,self.video_frames,B,x.size(-1)).permute(0,2,1,3).reshape(-1,BT,x.size(-1))
        x = x.permute(1, 0, 2)  					# LND -> NLD
        # Move the three lines below to `encode_image` for entire hidden sequence
        
        #x = self.ln_post(x[:, 0, :])
        #if self.proj is not None:
        #     x = x @ self.proj
        return x

        #return x, cluster_loss


class CLIP(nn.Module):
    def __init__(self,
                    embed_dim: int,
                    # vision
                    image_resolution: int,
                    vision_layers: Union[Tuple[int, int, int, int], int],
                    vision_width: int,
                    vision_patch_size: int,
                    # text
                    context_length: int,
                    vocab_size: int,
                    transformer_width: int,
                    transformer_heads: int,
                    transformer_layers: int,
                    # vision linear of patch
                    linear_patch: str = '2d',
                    video_frames=None,
                    args=None
                    ):
        """"
        Args:
            cluster: the number of cluster
        """
        super().__init__()

        self.context_length = context_length
        # visual encoder
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch,
                video_frames=video_frames,
                args=args
            )
        # text encoder
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        
        ## code for prompt transformer
        self.video_frames = video_frames
        self.unified_prompt_width = vision_width
        self.unified_text_prompt_length = 8
        self.unified_visual_prompt_length = 4*self.video_frames
        self.unified_prompt_length = self.unified_text_prompt_length+self.unified_visual_prompt_length
        self.unified_prompt_layers = transformer_layers
        self.unified_prompt_tokens = torch.arange(self.unified_prompt_length).long()
        self.unified_prompt_embedding = nn.Embedding(self.unified_prompt_length, self.unified_prompt_width*self.unified_prompt_layers)

        '''
        with torch.no_grad():
            for i in range(0,self.unified_prompt_width*self.unified_prompt_layers,self.unified_prompt_width):
                self.unified_prompt_embedding[:,i+8:i+9,:].data = self.visual.class_embedding[None,None,:].data
        self.prompt_embedding.requires_grad = True
        '''

        ##code for choice 2: visual/text output prompt token, initial with clip visual/text cls token

        self.unified_prompt_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(vision_width, vision_width * 2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(vision_width * 2, transformer_width))
        ]))
        PromptTransformer_heads =   self.unified_prompt_width//64
        self.PromptTransformer = PromptTransformer(
            width= self.unified_prompt_width,
            layers= 1,
            heads=PromptTransformer_heads)
        ### msrvtt words length= 32 4+32+4 =40
        self.text_prompt_length = 40
        self.text_prompt_prefix = self.unified_text_prompt_length//2
        self.text_prompt_dropout = nn.Dropout(0.0)
        
        ### code for multi layer text prompt
        '''
        self.text_prompt_prefix = 4
        
        self.text_hidden_size = 512
        self.text_prompt_layers = transformer_layers
        self.text_prompt_embedding = torch.nn.Embedding(self.text_prompt_length, self.text_hidden_size*self.text_prompt_layers)
        self.text_prefix_tokens = torch.arange(self.text_prompt_length).long()
        
        self.video_frames = video_frames
        #####
        '''
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        #text prompt_embedding init 
        #nn.init.normal_(self.text_prompt_embedding.weight, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def encode_prompt(self,batch_size,device):
        unified_prompt_tokens = self.unified_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        unified_prompt_embedding = self.unified_prompt_embedding(unified_prompt_tokens)
        unified_prompt_embedding = unified_prompt_embedding.view(batch_size,self.unified_prompt_length,self.unified_prompt_layers,self.unified_prompt_width)
        #nxlx(dx12) -> (nx12)xlxd -> lx(nx12)xd

        unified_prompt_embedding =  unified_prompt_embedding.permute(2,0,1,3)  ##layers,bz,length,width
        #print('unified_prompt_embedding.shape',unified_prompt_embedding.shape)
        #self.unified_prompt_layers,batch_size,self.unified_prompt_length,self.unified_prompt_width = unified_prompt_embedding.shape
        unified_prompt_embedding= unified_prompt_embedding.reshape(self.unified_prompt_layers*batch_size,self.unified_prompt_length,self.unified_prompt_width).permute(1,0,2)
     
        unified_prompt_output = self.PromptTransformer(unified_prompt_embedding)
        unified_prompt_output = unified_prompt_output.permute(1,0,2).view(self.unified_prompt_layers,batch_size,self.unified_prompt_length,self.unified_prompt_width)
        unified_text_prompt = self.unified_prompt_mlp(unified_prompt_output[:,:,:self.unified_text_prompt_length,:])
        unified_visual_prompt = unified_prompt_output[:,:,self.unified_text_prompt_length:,:]
        #unified_visual_prompt[:,:,0,:].data = self.visual.class_embedding.data
        return unified_text_prompt, unified_visual_prompt

    def encode_image(self, image,unified_visual_prompt,return_hidden=False,video_frame=-1):
        # hidden [N, L, D]
        
        #hidden, cluster_loss = self.visual(image.type(self.dtype), video_frame=video_frame)

        hidden = self.visual(image.type(self.dtype),unified_visual_prompt, video_frame=video_frame)
        hidden = self.visual.ln_post(hidden) @ self.visual.proj
        x = hidden[:, 0, :]
        #x_local = hidden[:,4:,:].reshape(hidden.size(0),-1,self.video_frames,hidden.size(-1)).permute(0,2,1,3).reshape(hidden.size(0)*self.video_frames,-1,hidden.size(-1))[:,0,:]
        #x =  x_global.repeat(self.video_frames,1)+x_local
        if return_hidden:
            return x, hidden

        return x 

    def encode_text_light(self, text):
        x_light = self.token_embedding(text).type(self.dtype)   # [batch_size, n_ctx, d_model]
        return x_light

    def encode_text_(self, xlight, text):
        x = xlight + self.positional_embedding[:xlight.size(1), :].type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def incorporate_text_prompt(self, x, prompt_embeddings,text):
        text_length = text.size(1)
        prompt_token = torch.zeros(x.size(0),self.text_prompt_length)

        #prompt_embeddings[:,0:1,:] = x[:,0:1,:]
        prompt_token[:,0:1] = text[:,0:1]

        #prompt_embeddings[:, self.text_prompt_prefix+1: self.text_prompt_prefix+ text_length,:] = x[:,1:text_length,:]
        prompt_token[:, self.text_prompt_prefix+1: self.text_prompt_prefix+text_length] = text[:,1:text_length]

        prompt_embeddings_output = torch.cat((x[:,0:1,:],
                            prompt_embeddings[:, :self.text_prompt_prefix,:],
                            x[:,1:text_length,:],
                            prompt_embeddings[:,self.text_prompt_prefix:,:],
                            ),dim=1)

        return prompt_embeddings_output,prompt_token

    def forward_deep_text_prompt(self, x, text_prompt_embedding,text_length):

        ## x.shape L,N,D (N=BxT)
        hidden_states = None
        B = x.shape[1]
        num_layers = self.unified_prompt_layers

        for i in range(num_layers):
            if i == 0:
                ##(cls_token + n_prompt + n_patches,batch_size, hidden_dim) (55,768,768)
                hidden_states = self.transformer.resblocks[i]((x,self.video_frames,False))[0]
            else:
                if i <= len(text_prompt_embedding):
                    # (768,5,768)
                    deep_prompt_emb = self.text_prompt_dropout(text_prompt_embedding[i])
                     
                    #NLD->LND to input transformer resblocks. #(5,768,768)
                    deep_prompt_emb = deep_prompt_emb.permute(1,0,2)

                    deep_prompt_emb = torch.cat((
                        hidden_states[:1,:, :],
                        deep_prompt_emb[:self.text_prompt_prefix,:,:],
                        hidden_states[self.text_prompt_prefix+1:self.text_prompt_prefix+text_length,:,:],
                        deep_prompt_emb[self.text_prompt_prefix:,:,:]
                    ), dim=0)

                    #deep_prompt_emb[0:1,:,:] = hidden_states[0:1,:,:]
                    #deep_prompt_emb[self.text_prompt_prefix+1:self.text_prompt_prefix+ text_length,:,:] = hidden_states[self.text_prompt_prefix+1:self.text_prompt_prefix+text_length,:,:]

                hidden_states = self.transformer.resblocks[i]((deep_prompt_emb,self.video_frames,False))[0]

        #    if self.transformer.vis:
        #        attn_weights.append(weights)
        #encoded = self.encoder.encoder_norm(hidden_states)
        return hidden_states

    def encode_text(self, text,unified_text_prompt, return_hidden=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]  
        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        
        ###add for text prompt
        batch_size = x.shape[0]
        #prefix_tokens = self.text_prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(text.device)
        #prompt_embeddings = self.text_prompt_embedding(prefix_tokens)
        '''
        prompt_embeddings = self.text_prompt_embedding(self.text_prefix_tokens.to(text.device))[None,:].repeat([batch_size,1,1])

        prompt_embeddings = prompt_embeddings.view(batch_size,self.text_prompt_length,self.text_prompt_layers,self.text_hidden_size)
        prompt_embeddings = self.text_prompt_dropout(prompt_embeddings)
        prompt_embeddings = prompt_embeddings.permute(2,0,1,3)
        '''
        text_length = text.shape[1]

        x,prompt_token = self.incorporate_text_prompt(x,unified_text_prompt[0],text)
        
        ###
        x = x.permute(1, 0, 2)  # NLD -> LND
        #x = self.transformer(x,self.video_frames,False)
        x = self.forward_deep_text_prompt(x,unified_text_prompt,text_length)
        x = x.permute(1, 0, 2)  # LND -> NLD
    
        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), prompt_token.argmax(dim=-1)]
        # x = torch.mean(hidden, dim=1)
        if return_hidden:
            return x, hidden

        # ORIGINAL CLIP CODE
        # x = self.ln_final(x).type(self.dtype)

        # # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        #image_features, cluster_loss = self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_clip_model(state_dict: dict, convert_fp16=True, linear_patch='2d', cut_top_layer=0,
                        load_state_dict=True, is_eval=True, 
                        video_frames=None,
                        args=None):
    """build a CLIP model
    Args:
        state_dict: the pretrained weights
        convert_fp16: If True, convert applicable model parameters to fp16
        linear_patch: the patch manner of image / video
        cut_top_layer: abandon a few top layers
        cluster: the number of cluster
        args: all the config arguments
    Return:
        A CLIP model, config of CLIP
    """
    clip_config = {}
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    # print info
    log_info("\n config of CLIP:\n"
                "\t embed_dim: {}\n"
                "\t image_resolution: {},\n"
                "\t vision_layers: {},\n"
                "\t vision_width: {},\n"
                "\t vision_patch_size: {},\n"
                "\t video_frames: {},\n"
                "\t context_length: {},\n"
                "\t vocab_size: {},\n"
                "\t transformer_width: {},\n"
                "\t transformer_heads: {},\n"
                "\t transformer_layers: {},\n".format(embed_dim, image_resolution, vision_layers,
                vision_width, vision_patch_size, video_frames,
                context_length, vocab_size, transformer_width,
                transformer_heads, transformer_layers))
    clip_config['context_length'] = context_length
    clip_config['transformer_width'] = transformer_width
    clip_config['transformer_heads'] = transformer_heads

    model = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads,
            transformer_layers - cut_top_layer,
            linear_patch=linear_patch, video_frames=video_frames, args=args
        ).float()

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    if convert_fp16:
        convert_weights(model)
    
    if load_state_dict:
        model.load_state_dict(state_dict)

    if is_eval:
        model.eval()

    return model, clip_config


##############################################################################
# utils for downloading CLIP pretrained weights and loading the pretrained state_dict
# https://github.com/openai/CLIP/blob/main/clip/clip.py
# 
##############################################################################

def load_clip_state_dict(pretrained_clip_name="ViT-B/32", pretrained_dir=os.path.expanduser("~/models/pretrained")):
    """load pretrained CLIP state dict from local file
    Args:
        pretrained_clip_name: name of pretrained CLIP model
        pretrained_dir: where the pretrained weight file located
    """
    if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
        model_path = os.path.join(pretrained_dir, _PT_NAME[pretrained_clip_name])
    else:
        raise NotImplementedError('Do not find CLIP model with name {}'.format(pretrained_clip_name))

    if pretrained_clip_name in ["ViT-B/32", "ViT-B/16"] and os.path.exists(model_path):
        pass
    else:
        raise IOError("Not found {}".format(model_path))
        if pretrained_clip_name in _MODELS:
            model_path = _download(_MODELS[pretrained_clip_name], root=pretrained_dir)
        elif os.path.isfile(pretrained_clip_name):
            model_path = pretrained_clip_name
        else:
            raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return state_dict


def _download(url: str, root: str = os.path.expanduser("~/models/pretrained")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())
