import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TRANSFORMER(nn.Module):
    def __init__(self, config):
        super(TRANSFORMER, self).__init__()
        self.config = config
        # self.num_feature = 2

        self.user_net = Net(config, 'user')
        self.item_net = Net(config, 'item')

    def forward(self, data):
        user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary = data
        user_all_summary, user_id, user2itemList = user_all_summary.to(self.config.device), user_id.to(self.config.device), user2itemList.to(self.config.device)
        item_all_summary, item_id, item2userList = item_all_summary.to(self.config.device), item_id.to(self.config.device), item2userList.to(self.config.device)

        user_feature = self.user_net(user_all_summary, user_id, user2itemList)
        item_feature = self.item_net(item_all_summary, item_id, item2userList)
        return user_feature, item_feature


class Net(nn.Module):
    def __init__(self, config, user_or_item='user'):
        super(Net, self).__init__()
        self.config = config

        if user_or_item == 'user':
            id_num = config.user_num
            ui_id_num = config.item_num
        elif user_or_item == 'item':
            id_num = config.item_num
            ui_id_num = config.user_num

        self.id_embedding = nn.Embedding(id_num, config.id_emb_dim)
        self.u_i_id_embedding = nn.Embedding(ui_id_num, config.id_emb_dim)
        # self.summary_embedding = nn.Embedding(config.vocab_size, config.summary_dim)

        self.transformersLayer = TransformerLayer(config)

        self.summary_linear = nn.Linear(config.feature_dim, config.id_emb_dim)
        self.id_linear = nn.Linear(config.id_emb_dim, config.id_emb_dim, bias=False)
        self.attention_linear = nn.Linear(config.id_emb_dim, 1)
        self.fc_layer = nn.Linear(config.feature_dim, config.id_emb_dim)

        self.dropout = nn.Dropout(config.drop_out)
        self.init_param()

    def init_param(self):
        # nn.init.xavier_normal_(self.transformersLayer.weight)
        # nn.init.constant_(self.transformersLayer.bias, 0.1)

        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.summary_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.summary_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)

        nn.init.uniform_(self.id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-0.1, b=0.1)

    def forward(self, summary, ids, ids_list):
        """
        b = 1
        :param summary:  [b, seq_num, sep_len]
        :param ids:  [b, 1]
        :param ids_list:  [b, seq_num]
        :return:
        """
        # # --------------- word embedding ----------------------------------
        # summary = self.summary_embedding(summary)
        # bs, num, seq_len, wd = summary.size()
        # summary = summary.view(-1, seq_len, wd)
        bs, num, seq_len = summary.size()

        # ids:  [1, dim]
        id_emb = self.id_embedding(ids).squeeze(1)

        # uiid_emb_output:  [1, seq_num, dim]
        uiid_emb_output = self.u_i_id_embedding(ids_list)

        # --------transformers for summary--------------------

        # feature: [seq_num, seq_len, dim]
        feature = F.relu(self.transformersLayer(summary)).transpose(1,2)

        # feature: [seq_num, dim]
        feature = F.max_pool1d(feature, feature.size(2)).squeeze(2)

        # feature: [1, seq_num, dim]
        feature = feature.view(-1, num, feature.size(1))

        # ------------------linear attention-------------------------------
        # summary_linear_output: [1, seq_num, dim]
        summary_linear_output = self.summary_linear(feature)

        # id_linear_output: [1, seq_num, dim]
        id_linear_output = self.id_linear(F.relu(uiid_emb_output))

        # rs_mix: [1, seq_num, dim]
        rs_mix = F.relu(summary_linear_output + id_linear_output)

        # att_score: [1, seq_num, 1]
        att_score = self.attention_linear(rs_mix)
        # att_score: [1, seq_num, 1]
        att_weight = F.softmax(att_score, 1)

        # summary_feature_output: [1, seq_num, dim]
        summary_feature = feature * att_weight
        # summary_feature_output: [1, dim]
        summary_feature = summary_feature.sum(dim=1)
        summary_feature = self.dropout(summary_feature)
        # # summary_feature_output: [1, dim]
        # summary_feature_output = self.fc_layer(summary_feature)

        return torch.stack([id_emb, summary_feature], dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, config, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout()
        self.config = config

        pe = torch.zeros(max_len, config.summary_dim).to(config.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.summary_dim, 2).float() * (-math.log(10000.0) / config.summary_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x.to(self.config.device)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.config = config

        self.Embedding = nn.Embedding(config.vocab_size, config.feature_dim)
        self.posEmb = PositionalEncoding(config)
        self.transformer = nn.Transformer(config.feature_dim, num_encoder_layers=6, num_decoder_layers=6, batch_first=True)

    def init_params(self):
        nn.init.xavier_normal_(self.summary_embedding.weight)

    def forward(self, summary):
        """
        b = 1
        :param summary: [b, seq_num, seq_len]
        :return:
        """
        #[seq_num, seq_len]
        src, tgt = summary.squeeze(0), summary.squeeze(0)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).bool().to(self.config.device)
        srcPaddingMask = TransformerLayer.getPaddingMask(src).to(self.config.device)
        tgtPaddingMask = TransformerLayer.getPaddingMask(tgt).to(self.config.device)

        # [seq_num, seq_len, wd]
        src = self.posEmb(self.Embedding(src))
        tgt = self.posEmb(self.Embedding(tgt))

        # [seq_num, seq_len, wd]
        output = self.transformer(src, tgt, tgt_mask=tgt_mask,
                                  src_key_padding_mask=srcPaddingMask, tgt_key_padding_mask=tgtPaddingMask)
        return output

    @staticmethod
    def getPaddingMask(tokens):
        """
        用于padding_mask
        """
        paddingMask = torch.zeros(tokens.size())
        paddingMask[tokens == 100] = -1e9
        return paddingMask
