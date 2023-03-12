import torch
from .fuse_helper import BiAttentionBlockForCheckpoint
import torch.utils.checkpoint as checkpoint
# from transformers.models.bert.modeling_bert import BertConfig, BertAttention, BertIntermediate, BertOutput, \
#     BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
from .modeling_bert import BertAttention, BertIntermediate, BertOutput

class BertEncoderLayer(BertPreTrainedModel):
    def __init__(self, config,  clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super().__init__(config)
        self.config = config

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        

        self.attention = BertAttention(config,  clamp_min_for_underflow, clamp_max_for_overflow)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        hidden_states = language_dict_features["hidden"]
        attention_mask = language_dict_features["masks"]

        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        self_attention_outputs = self.attention(
            hidden_states,
            extended_attention_mask,
            None,
            output_attentions=False,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        hidden_states = outputs[0]

        language_dict_features["hidden"] = hidden_states

        features_dict = {"visual": inputs["visual"],
                         "lang": language_dict_features
                         }

        return features_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self, cfg):
        super(VLFuse, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg
        self.use_checkpoint = cfg.MODEL.VL_FUSION_USE_CHECKPOINT

        # early fusion module
        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlockForCheckpoint(v_dim=self.img_dim, # 256
                    l_dim=self.lang_dim, # 768
                    embed_dim=self.embed_dim, # 2048
                    num_heads=self.n_head, # 8
                    dropout=0.1,
                    drop_path=.0,
                    init_values=1.0 / cfg.MODEL.DDETRS.ENC_LAYERS,
                    cfg=cfg
                    )
    def init_configs(self, cfg):
        # common params
        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.img_dim = cfg.MODEL.DDETRS.HIDDEN_DIM # 256

        self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        self.n_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS

        # mha params
        self.n_head = 8
        self.embed_dim = cfg.MODEL.DDETRS.VL_HIDDEN_DIM # 2048 by default

        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM # 768
        else:
            self.lang_dim = 1024

    def forward(self, x, task=None):
        visual_features = x["visual"]
        language_dict_features = x["lang"]

        if self.use_checkpoint:
            fused_visual_features, language_features = checkpoint.checkpoint(self.b_attn,
                visual_features, language_dict_features['hidden'], language_dict_features['masks'], task)
        else:
            fused_visual_features, language_features = self.b_attn(
                visual_features, language_dict_features['hidden'], language_dict_features['masks'], task)

        language_dict_features['hidden'] = language_features
        fused_language_dict_features = language_dict_features

        features_dict = {"visual": fused_visual_features,
                         "lang": fused_language_dict_features}

        return features_dict
