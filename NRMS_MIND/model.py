import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from layers import SelfAttn, LinearAttn

class Encoder(nn.Module):
    def __init__(self, drop, input_dim, output_dim, key_dim, head_num, head_dim):
        super(Encoder, self).__init__()
        # self.dropout = nn.Dropout(p=drop)
        self.self_attn = SelfAttn(head_num=head_num,
                                  head_dim=head_dim,
                                  input_dim=input_dim)
        self.linear_attn = LinearAttn(output_dim=output_dim, key_dim=key_dim)
    def forward(self, **kwargs):
        pass

class NewsEncoder(Encoder):
    def __init__(self, word_emb, drop, word_dim, news_dim, key_dim, head_num, head_dim):
        super(NewsEncoder, self).__init__(drop=drop, input_dim=word_dim,
                                          output_dim=news_dim, key_dim=key_dim,
                                          head_num=head_num, head_dim=head_dim)
        self.word_emb = word_emb

    def forward(self,inputs):
        # print("inputs")
        # print(inputs.shape)
        inputs = self.word_emb(inputs)
        # print("outputs")
        # print(inputs.shape)
        # out = self.dropout(x)
        inputs = inputs.float()
        # print(inputs.shape)
        out = self.self_attn(QKV=(inputs,inputs,inputs))
        # out = self.dropout(out)
        out = self.linear_attn(out)
        return out

class UserEncoder(Encoder):
    def __init__(self, news_encoder, drop, news_dim, user_dim, key_dim, head_num, head_dim):
        super(UserEncoder, self).__init__(drop=drop, input_dim=news_dim,
                                          output_dim=user_dim, key_dim=key_dim,
                                          head_num=head_num, head_dim=head_dim)
        self.news_encoder = news_encoder

    def forward(self, x):  # x : [32, 30] torch tensor
        x = self.news_encoder(x)
        # out = self.dropout(x)
        # out = self.self_attn(QKV=(out, out, out))
        # out = self.dropout(out)
        # out = self.linear_attn(out)
        out = self.self_attn(QKV=(x, x, x))
        out = self.linear_attn(out)
        return out

class NRMS(nn.Module):
    # input : trn_his or trn_cand (torch tensor), word embedding file (npy)
    # do :
    # output : torch tensor
    def __init__(self, hparams, word2vec_embedding):
        super(NRMS, self).__init__()
        self.word_emb = nn.Embedding(word2vec_embedding.shape[0], hparams['word_emb_dim'])
        self.word_emb.weight = nn.Parameter(torch.tensor(word2vec_embedding, dtype=torch.float32))
        self.word_emb.weight.requires_grad = True
        # print(self.word_emb.weight.grad)
        self.news_dim = hparams['head_num'] * hparams['head_dim'] # 20*20 = 400
        self.user_dim = self.news_dim # 400
        self.key_dim = hparams['attention_hidden_dim'] # 200

        self.news_encoder = NewsEncoder(word_emb=self.word_emb,
                                        drop=hparams['dropout'],
                                        word_dim=hparams['word_emb_dim'],
                                        news_dim=self.news_dim,
                                        key_dim=self.key_dim,
                                        head_num=hparams['head_num'],
                                        head_dim=hparams['head_dim'])
        self.user_encoder = UserEncoder(news_encoder=self.news_encoder,
                                        drop=hparams['dropout'],
                                        news_dim=self.news_dim,
                                        user_dim=self.user_dim,
                                        key_dim=self.key_dim,
                                        head_num=hparams['head_num'],
                                        head_dim=hparams['head_dim'])

    def forward(self, input, source):
        if source == "history":
            # print("history")
            his_out = self.user_encoder(input)
            return his_out
        elif source == "candidate":
            # print("candidate")
            cand_out = self.news_encoder(input)
            return cand_out