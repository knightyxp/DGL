# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from .losses import CrossEn
from .base import PreTrainedModel
from .module_cross import CrossConfig, CrossModel
from .utils import all_gather, log_info, update_attr
from .clip import build_clip_model, load_clip_state_dict
from .module_cross import Transformer as TransformerClip
##add for efficient prompt
from collections import OrderedDict
import einops

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        """"create a model from pretrained config or model weights files"""
        task_config = kwargs['task_config']
        if state_dict is None: state_dict = {}

        # load pretrained CLIP state_dict
        pretrained_clip_name = getattr(task_config, 'pretrained_clip_name', "ViT-B/32") 
        clip_state_dict = load_clip_state_dict(pretrained_clip_name, pretrained_dir=task_config.pretrained_dir)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size,
                                                    state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        # divide the pretrained temperature
        # model.clip.logit_scale.data.fill_(task_config.temperature_new)
        if task_config.temperature_new > 1.0:
            logging.info("Assign new temperature {} to the logit_scale".format(task_config.temperature_new))
            model.clip.logit_scale.data.fill_(task_config.temperature_new)
        
        return model


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        """
        A CLIP4Clip model in https://github.com/ArrowLuo/CLIP4Clip
        Args:
            cross_config: config arguments of cross_model
            clip_state_dcit: the weight state_dict of pretrained clip
            task_config: config args
        """
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        # assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
        self._stage_one = True
        self._stage_two = False
        # tightTransf or not
        self.loose_type = True if (self._stage_one and task_config.loose_type) else False
        # patch manner of images / video
        self.linear_patch = getattr(task_config, "linear_patch", '2d')
        self.sim_header = getattr(task_config, "sim_header", 'meanP')
        if self.sim_header == "tightTransf": assert self.loose_type is False

        self.video_frames = getattr(task_config, "max_frames", None)
        self.time_embedding = getattr(task_config, "time_embedding", None)
        self.freeze_clip = getattr(task_config, "freeze_clip", 0)
        self.new_added_modules = getattr(task_config, "new_added_modules", [None, ])
        self.pre_visual_pooling = getattr(task_config, "pre_visual_pooling", 0)
        # https://github.com/starmemda/CAMoE/blob/main/DSL.py
        self.camoe_dsl = getattr(task_config, "camoe_dsl", False)
        ##add for efficient prompt
        self.tfm_layers = getattr(task_config,'tfm_layers',None)
        self.tfm_heads = getattr(task_config,'tfm_heads',None)
        self.dropout = 0.0
        log_info("\n config of CLIP4Clip:\n"
                    "\t Stage-One:{}\n"
                    "\t Stage-Two:{}\n"
                    "\t loose type {}\n"
                    "\t linear_patch: {}\n"
                    "\t sim_header: {}\n"
                    "\t camoe_dsl: {}\n".format(self._stage_one, self._stage_two, self.loose_type,
                                                self.linear_patch, self.sim_header, self.camoe_dsl))
        # create CLIP Encoders
        self.clip, clip_config = build_clip_model(clip_state_dict, convert_fp16=True,
                                                    linear_patch=self.linear_patch,
                                                    cut_top_layer=0, load_state_dict=False,
                                                    is_eval=False,
                                                    video_frames=self.video_frames,
                                                    args=task_config)

        cross_config.max_position_embeddings = clip_config['context_length']

        if self.loose_type is False:
            # Tight types transformer for similarity calculator
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=clip_config['transformer_width'],
                                                    layers=self.task_config.cross_num_hidden_layers,
                                                    heads=clip_config['transformer_heads'])

        self.loss_fct = CrossEn()
        
        # code of CLIP4Clip, will this broke the pretrained parameters of CLIP ?
        # self.apply(self.init_weights)

        ## add for prompt
    '''   
        self.temporal = 1
    
        self.hidden_size = 512
        self.embedding = torch.nn.Embedding(77, self.hidden_size)
        self.temporalEmbedding = torch.nn.Embedding(self.video_frames, self.hidden_size)

        if self.temporal == 1:
           self.temporalModelling = TemporalModelling(width=self.hidden_size, layers=self.tfm_layers, heads=self.tfm_heads, dropout=self.dropout)
        self.prefix = 4
        self.initialize_parameters()
        

    def initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)
    '''       
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, video=None, video_mask=None,
                    pre_visual_pooling=False):
        """
        Args:
            input_ids: text input ids, [N, L]
            token_type_ids: token type ids, [N, L]
            attention_mask: [N, L]
            video: video tensor
            video_mask: video mask
        """
        output_dict = {'sequence_output': None,
                        'visual_output': None,
                        'loss': None
                        }
        
        if input_ids is not None and video is not None: 
            batch_size = input_ids.size(0)
            unified_text_prompt,unified_visual_prompt = self.clip.encode_prompt(batch_size,device=input_ids.device)
        
        if input_ids is not None:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

            sequence_output = self.get_sequence_output(input_ids, unified_text_prompt,token_type_ids, attention_mask)
            output_dict['sequence_output'] = sequence_output

        if video is not None:
            # T x 3 x H x W
            # convert to float
            video = torch.as_tensor(video).float()
            # b, pair, video_frame, channel, h, w = video.shape
            # video = video.view(b * pair * bs * ts, channel, h, w)
            # video_frame = bs * ts
            batch_size, pair, video_frame, channel, h, w = video.shape
            #print('initial_video_mask:',video_mask.shape)

            video_mask = video_mask.view(-1, video_mask.shape[-1])
            visual_output = self.get_visual_output(video,unified_visual_prompt,video_mask, video_frame=video_frame) 
            if self.training or not self.pre_visual_pooling:
                output_dict['visual_output'] = visual_output
            else:
                visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
                visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
                visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
                output_dict['visual_output'] = visual_output

        if self.training:
            ## org 
            loss = 0.
            sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, visual_output,
                                                            attention_mask, video_mask,
                                                            shaped=True)
            # contrastive loss
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss
          
            output_dict['loss'] = loss

            output_dict['sim_loss'] = sim_loss

        return output_dict

    def get_sequence_output(self, input_ids,unified_text_prompt,token_type_ids=None, attention_mask=None):
        """text encoding"""
        bs_pair = input_ids.size(0)
        # encode text and convert to float
        sequence_hidden = self.clip.encode_text(input_ids,unified_text_prompt).float()
        tFeature = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        
        # [bs_pair, 1, D]
        return tFeature

    def get_visual_output(self,video,unified_visual_prompt,video_mask=None, video_frame=-1):
        """image encoding"""
        bs_pair = video_mask.size(0)
        
        batch_size, pair, video_frame, channel, h, w = video.shape
        
        '''
        
        iFeature = self.clip.encode_image(einops.rearrange(video, 'b p t c h w -> (b p t) c h w'),video_frame = video_frame)
        #p = 1
        vFeature = einops.rearrange(iFeature, '(b t) c -> t b c', t=video_frame)
    
        if self.temporal == 1:
            tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(video_frame).to(vFeature.device)), 't c -> t b c', b=batch_size)
            vFeature = vFeature + tempEmbedding
            vFeature = self.temporalModelling(vFeature)  
            vFeature = vFeature.mean(dim=0)
        else:
            vFeature = vFeature.type(torch.float32).mean(dim=0)
        #print('vFeature',vFeature.shape)
        vFeature = vFeature.view(bs_pair,-1,vFeature.size(-1))
        #visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        #print('visual_hidden:',visual_hidden.shape)
        '''
        video = video.view(-1, channel, h, w)
        visual_hidden = self.clip.encode_image(video,unified_visual_prompt,video_frame=video_frame)
        vFeature = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        
        # [bs_pair, 1, D]
        return vFeature


    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
        # if (self.cluster is not None and self.cluster > 1) or self.cluster_inter or self.deep_cluster:
            # video_out = torch.mean(visual_output, dim=1)
        # else:
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        # avoid zero divide
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if self.sim_header == "meanP":
            # Default: Parameter-free type
            pass

        elif self.sim_header == "seqLSTM":
            raise NotImplementedError

        elif self.sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = all_gather(visual_output)
            video_mask = all_gather(video_mask)
            sequence_output = all_gather(sequence_output)
            torch.distributed.barrier()

        if self.training or not self.pre_visual_pooling:
            visual_output = visual_output.squeeze(1)
            #visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
            #visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
            visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1).to(device=attention_mask.device,
                            dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if self.loose_type:
            assert self.sim_header in ["meanP", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask)
        else:
            assert self.sim_header in ["tightTransf"]
            raise NotImplementedError
            # retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)

        # if self.camoe_dsl:
        # 	# https://github.com/starmemda/CAMoE
        # 	retrieve_logits = retrieve_logits * F.softmax(retrieve_logits, dim=0) * len(retrieve_logits)

        return retrieve_logits, contrastive_direction
    '''
    def freeze_cip_layers(self, freeze_layer_num):
        """freeze the parameters of layers with No.layer < freeze_layer_num"""
        ## ####################################
        # freeze testing
        ## ####################################
        assert freeze_layer_num <= 12 and freeze_layer_num >= -1
        if hasattr(self, "clip") and freeze_layer_num > -1:
            for name, param in self.clip.named_parameters():
                # top layers always need to train
                if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                        or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                    continue
                # top resblocks or transformer layers need to train
                elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                    layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                    if layer_num >= freeze_layer_num:
                        continue    # need to train

                if self.linear_patch == "3d" and name.find("conv2."):
                    continue
                else:
                    # paramenters which < freeze_layer_num will be freezed
                    param.requires_grad = False
    '''
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(CLIP4Clip, self).train(mode)
        no_clip = self.new_added_modules

        if self.freeze_clip and mode:
            logging.info("(model) Freezing ALL the CLIP backbone.")
            for name, param in self.clip.named_parameters():
                if not any(nd in name for nd in no_clip):
                    param.requires_grad = False
                else:
                    logging.info('trainerble parameters are:{}'.format(name))

            
            for name, m in self.clip.named_modules():
                if not any(nd in name for nd in no_clip):
                    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.Dropout)):
                        m.eval()
            for m in self.clip.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()

    def print_trainable_params_percentage(self):
        # if "bart-base" in self.args.backbone:
        #     orig_param_size = 139420416
        # elif "t5-base" in self.args.backbone:
        #     orig_param_size = 222903552
        # else:
        #     print(f"Don't know the parameters number of this {self.args.backbone}")
        #     orig_param_size = -1

        orig_param_size = sum(p.numel() for p in self.clip.parameters())

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainable_size = count_parameters(self.clip)

        percentage = trainable_size / orig_param_size * 100

        print(f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})")

        return percentage