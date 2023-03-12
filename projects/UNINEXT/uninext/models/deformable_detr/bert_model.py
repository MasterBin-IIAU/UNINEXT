from copy import deepcopy
import numpy as np
import torch
from torch import nn

# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel


class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()
        self.cfg = cfg
        self.bert_name = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE

        if self.bert_name == "bert-base-uncased":
            config = BertConfig.from_pretrained("projects/UNINEXT/%s"%self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = BertModel.from_pretrained("projects/UNINEXT/%s" % self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        elif self.bert_name == "roberta-base":
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        else:
            raise NotImplementedError

        self.num_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS # 1
        self.parallel_det = cfg.MODEL.PARALLEL_DET

    def forward(self, x, task=None):
        input = x["input_ids"] # (bs, seq_len)
        mask = x["attention_mask"] # (bs, seq_len)

        if self.parallel_det and task == "detection":
            # disable interaction among tokens
            bs, seq_len = mask.shape
            mask_new = torch.zeros((bs, seq_len, seq_len), device=mask.device)
            for _ in range(bs):
                mask_new[_, :, :] = mask[_]
                num_valid = torch.sum(mask[_])
                mask_new[_, :num_valid, :num_valid] = torch.eye(num_valid)
            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask_new,
                output_hidden_states=True,
            )
        else:
            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask,
                output_hidden_states=True,
            )
        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        # features = None
        # features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1) # (bs, seq_len, language_dim)

        # # language embedding has shape [len(phrase), seq_len, language_dim]
        # features = features / self.num_layers

        # embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        # aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            # "aggregate": aggregate,
            # "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret
