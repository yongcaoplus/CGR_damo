# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 11:26 AM 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
from transformers import BertConfig, BertModel
import torch
import logging
from torch import nn
import torch.nn.functional as F


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        bert_config = BertConfig.from_pretrained(text_encoder)
        bert_config.fusion_layer = bert_config.num_hidden_layers
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)
        text_width = self.text_encoder.config.hidden_size
        self.embed_dim = config['embed_dim']
        self.exper_mode = config['exper_mode']

        self.main_avg_loss = []
        self.em_hera_avg_loss = []
        self.final_avg_loss = []
        self.query_proj = nn.Linear(text_width, config['embed_dim'])
        self.doc_proj = nn.Linear(text_width, config['embed_dim'])
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        logging.info("=========> We are on {} mode!".format(self.exper_mode))

    def gen_token_average_embs(self, embeddings, attention_masks):
        embs = attention_masks[:, 1:].unsqueeze(2).repeat(1, 1, embeddings.shape[2]) * embeddings[:, 1:, :]
        output_embs = embs.sum(1) / (attention_masks.sum(dim=1) - 1).unsqueeze(1).repeat(1, embs.shape[2])
        return output_embs

    def gen_token_average_embs_with_span(self, embeddings, attention_masks, key_span_mask, device):
        key_span_mask = key_span_mask.to(device)
        span_lens = key_span_mask.sum(dim=1) - 1
        span_embs = []
        for i, span_len in enumerate(span_lens):
            if span_len <= 0:
                emb = (embeddings[i, 1:, :] * attention_masks[i, 1:].unsqueeze(1).repeat(1, 1,
                                                                                         embeddings.shape[2])).sum(
                    dim=1) / (attention_masks[i, 1:].sum() - 1)
            else:
                emb = (key_span_mask[i, 1:].unsqueeze(1).repeat(1, 1, embeddings.shape[2]) * embeddings[i, 1:, :]).sum(
                    dim=1) / (key_span_mask[i, 1:].sum() - 1)
            span_embs.append(emb)
        span_embs = torch.cat(span_embs, dim=0)
        return span_embs

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1))  # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v)  # [bs, poly_m, dim]
        return output

    def dot_attention_2d(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(1, 0))  # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v)  # [bs, poly_m, dim]
        return output

    def gen_query_token_cls_embs(self, embeddings, attention_masks, cand_embeddings, cand_attention_masks):
        bs, _, emb_dim = embeddings.shape
        embs = attention_masks[:, 1:].unsqueeze(2).repeat(1, 1, emb_dim) * embeddings[:, 1:, :]
        cand_embs = cand_attention_masks[:, 1:].unsqueeze(2).repeat(1, 1, emb_dim) * cand_embeddings[:, 1:, :]
        cand_output_embs = cand_embs.sum(1) / (cand_attention_masks.sum(dim=1) - 1).unsqueeze(1).repeat(1, emb_dim)
        # cand_output_embs = cand_output_embs.reshape(bs, -1, emb_dim)
        # output_embs = self.dot_attention(embs, cand_output_embs, cand_output_embs)
        output_embs = embs.sum(1) / (attention_masks.sum(dim=1) - 1).unsqueeze(1).repeat(1, emb_dim)
        return output_embs

    def query_co_occur_cal(self, query_id, cand_id):
        co_occur_weight = torch.zeros_like(query_id)
        cand_id = cand_id.reshape(query_id.shape[0], -1, cand_id.shape[1])
        for i in range(cand_id.shape[1]):
            # optimization TODO
            case_weight = torch.zeros_like(query_id)
            for query_i in range(query_id.shape[0]):
                for query_j in range(query_id.shape[1]):
                    case_weight[query_i][query_j] = 1 if query_id[query_i][query_j] in cand_id[:, i, :] and \
                                                         query_id[query_i][
                                                             query_j] not in self.bert_special_token_ids else 0
            co_occur_weight += case_weight
        return co_occur_weight

    def cand_occur_weight(self, query_id, cand_id):
        cand_id = cand_id.reshape(query_id.shape[0], -1, cand_id.shape[1])
        co_occur_weight = torch.zeros_like(cand_id)
        # optimization TODO
        for i in range(cand_id.shape[0]):
            for j in range(cand_id.shape[1]):
                for k in range(cand_id.shape[2]):
                    co_occur_weight[i][j][k] = 1 if cand_id[i][j][k] in query_id[i] and cand_id[i][j][
                        k] not in self.bert_special_token_ids else 0
        co_occur_weight = co_occur_weight.reshape(-1, cand_id.shape[2])
        return co_occur_weight

    def shape_attention(self, query_id, query_attention_masks, cand_id, cand_attention_masks):
        bs, query_max_len = query_id.shape
        # query attention mask
        # import pdb;pdb.set_trace()
        query_attention_masks = self.sen_length_attn.repeat(bs, 1) * query_attention_masks
        query_cooccur_weight = self.query_co_occur_cal(query_id, cand_id)
        query_attention_masks += self.query_weight * query_cooccur_weight
        # cand attention mask
        cand_cooccur_weight = self.cand_occur_weight(query_id, cand_id)
        cand_attention_masks = cand_attention_masks.to(
            cand_cooccur_weight.dtype) + self.cand_weight * cand_cooccur_weight
        return query_attention_masks, cand_attention_masks

    def get_diff_embeds(self, embeds, mask=None):
        if self.exper_mode == "baseline_tok":
            mask = mask.unsqueeze(-1).expand(-1, -1, embeds.last_hidden_state.size(-1))
            breakpoint()
            return embeds[0]
        else:
            return embeds.last_hidden_state[:, 0, :]

    def forward(self, query, doc, bs, docnum):
        loss_constrain = 0.0
        query_output = self.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                         return_dict=True)
        query_embeds = self.get_diff_embeds(query_output, query.attention_mask)
        query_feat = self.query_proj(query_embeds)
        doc_output = self.text_encoder(doc.input_ids, attention_mask=doc.attention_mask,
                                       return_dict=True)
        doc_embeds = self.get_diff_embeds(doc_output, doc.attention_mask)
        doc_feat = self.doc_proj(doc_embeds)
        scores = query_feat.view(bs, 1, self.embed_dim).matmul(
            doc_feat.view(bs, docnum, self.embed_dim).transpose(1, 2))
        labels = torch.zeros(bs, dtype=torch.long, device=doc_feat.device)
        loss = self.cross_entropy_loss(scores.view(bs, docnum), labels)
        return loss, loss_constrain

    def constrain_loss(self, match_score, left_sim, right_sim):
        cons_loss = 0.0
        for i in range(len(match_score)):
            if match_score[i] > 0:
                if left_sim[i][i] < right_sim[i][i]:
                    cons_loss += 1.0
            else:
                if left_sim[i][i] > right_sim[i][i]:
                    cons_loss += 1.0
        return torch.tensor(cons_loss)

    def em_match_score(self, query_ids, doc_ids, left_index, doc_length):
        doc_num = int(doc_ids.shape[0] / query_ids.shape[0])
        left_ids, right_ids = [], []
        for item in range(doc_ids.shape[0]):
            left_ids.append(doc_ids[item][1:left_index[item]])
            right_ids.append(doc_ids[item][left_index[item]:doc_length[item]])
        ret = []
        for i in range(query_ids.shape[0]):
            left_id = torch.cat(left_ids[i * doc_num:(i + 1) * doc_num])
            right_id = torch.cat(right_ids[i * doc_num:(i + 1) * doc_num])
            query_id = query_ids[i][1:torch.sum(query_ids[i] != 0, dim=0)]
            left_score = torch.sum(torch.tensor([torch.sum(left_id == item) for item in query_id]))
            right_score = torch.sum(torch.tensor([torch.sum(right_id == item) for item in query_id]))
            if left_score >= right_score:
                ret.append(1)
            else:
                ret.append(0)
        return ret

    def match_score(self, query_ids, doc_ids):
        doc_num = int(doc_ids.shape[0] / query_ids.shape[0])
        left_index = (torch.sum(doc_ids[:, 1:] != 0, dim=1) / 2).to(torch.int)
        doc_length = torch.sum(doc_ids[:, 1:] != 0, dim=1)
        left_ids, right_ids = [], []
        for item in range(doc_ids.shape[0]):
            left_ids.append(doc_ids[item][1:left_index[item]])
            right_ids.append(doc_ids[item][left_index[item]:doc_length[item]])
        ret = []
        for i in range(query_ids.shape[0]):
            left_id = torch.cat(left_ids[i * doc_num:(i + 1) * doc_num])
            right_id = torch.cat(right_ids[i * doc_num:(i + 1) * doc_num])
            query_id = query_ids[i][1:torch.sum(query_ids[i] != 0, dim=0)]
            left_score = torch.sum(torch.tensor([torch.sum(left_id == item) for item in query_id]))
            right_score = torch.sum(torch.tensor([torch.sum(right_id == item) for item in query_id]))
            if left_score >= right_score:
                ret.append(1)
            else:
                ret.append(0)
        return ret, left_index, doc_length

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

