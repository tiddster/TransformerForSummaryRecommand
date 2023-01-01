# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DeepCoNN(nn.Module):
    '''
    deep conn 2017
    '''
    def __init__(self, config, uori='user'):
        super(DeepCoNN, self).__init__()
        self.config = config
        self.config.num_feature = 1
        self.config.predictionLayerType = 'fm'

        self.user_word_embs = nn.Embedding(config.vocab_size, config.word_dim)
        self.item_word_embs = nn.Embedding(config.vocab_size, config.word_dim)

        self.user_cnn = nn.Conv2d(1, config.filters_num, (config.kernel_size, config.summary_dim))
        self.item_cnn = nn.Conv2d(1, config.filters_num, (config.kernel_size, config.summary_dim))

        self.user_fc_linear = nn.Linear(config.filters_num, config.feature_dim)
        self.item_fc_linear = nn.Linear(config.filters_num, config.feature_dim)
        self.dropout = nn.Dropout(self.config.drop_out)

        self.init_param()

    def forward(self, datas):
        # summary:  [b, seq_num, sep_len]   b = 1
        user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary = datas

        user_all_summary = user_all_summary.squeeze(0)
        item_all_summary = item_all_summary.squeeze(0)

        # user_doc: [1, seq_num * sep_len, dim]
        user_doc = self.user_word_embs(user_all_summary).view(1, user_all_summary.shape[0] * user_all_summary[1], -1)
        item_doc = self.item_word_embs(item_all_summary).view(1, item_all_summary.shape[0] * item_all_summary[1], -1)

        # feature: [1, filters_num, seq_num * seq_len]
        u_fea = F.relu(self.user_cnn(user_doc.unsqueeze(1))).squeeze(3)
        i_fea = F.relu(self.item_cnn(item_doc.unsqueeze(1))).squeeze(3)
        # feature: [1, filter_num]
        u_fea = F.max_pool1d(u_fea, u_fea.size(2)).squeeze(2)
        i_fea = F.max_pool1d(i_fea, i_fea.size(2)).squeeze(2)
        # feature: [1, dim]
        u_fea = self.dropout(self.user_fc_linear(u_fea))
        i_fea = self.dropout(self.item_fc_linear(i_fea))

        return torch.stack([u_fea], dim=1), torch.stack([i_fea], dim=1)

    def init_param(self):

        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        # if self.config.use_word_embedding:
        #     w2v = torch.from_numpy(np.load(self.config.w2v_path))
        #     if self.config.use_gpu:
        #         self.user_word_embs.weight.data.copy_(w2v.cuda())
        #         self.item_word_embs.weight.data.copy_(w2v.cuda())
        #     else:
        #         self.user_word_embs.weight.data.copy_(w2v)
        #         self.item_word_embs.weight.data.copy_(w2v)
        # else:
        nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)