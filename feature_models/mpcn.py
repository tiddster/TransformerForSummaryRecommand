# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MPCN(nn.Module):
    '''
    Multi-Pointer Co-Attention Network for Recommendation
    WWW 2018
    '''
    def __init__(self, config, head=3):
        '''
        head: the number of pointers
        '''
        super(MPCN, self).__init__()

        self.config = config
        config.num_feature = 1
        self.head = config.num_heads

        self.user_summary_embs = nn.Embedding(config.vocab_size, config.summary_dim)
        self.item_summary_embs = nn.Embedding(config.vocab_size, config.summary_dim)

        # review gate
        self.fc_gate1 = nn.Linear(config.summary_dim, config.summary_dim)
        self.fc_gate2 = nn.Linear(config.summary_dim, config.summary_dim)

        # multi points
        self.summary_coatt = nn.ModuleList([Co_Attention(config.summary_dim, gumbel=True, pooling='max') for _ in range(config.num_heads)])
        self.word_coatt = nn.ModuleList([Co_Attention(config.summary_dim, gumbel=False, pooling='avg') for _ in range(config.num_heads)])

        # final fc
        self.fc_user = self.fc_layer()
        self.fc_item = self.fc_layer()

        self.drop_out = nn.Dropout(config.drop_out)
        self.init_params()

    def fc_layer(self):
        return nn.Sequential(
            nn.Linear(self.config.summary_dim * self.head, self.config.summary_dim),
            nn.ReLU(),
            nn.Linear(self.config.summary_dim, self.config.id_emb_dim)
        )

    def forward(self, datas):
        '''
        :user_all_summary: B * L1 * N
        :item_all_summary: B * L2 * N
        '''
        user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary = datas
        user_all_summary, user_id, user2itemList = user_all_summary.to(self.config.device), user_id.to(
            self.config.device), user2itemList.to(self.config.device)
        item_all_summary, item_id, item2userList = item_all_summary.to(self.config.device), item_id.to(
            self.config.device), item2userList.to(self.config.device)

        # ------------------summary-level co-attention ---------------------------------
        # [1, num, dim]
        user_word_embs_output = self.user_summary_embs(user_all_summary)
        item_word_embs_output = self.item_summary_embs(item_all_summary)
        # [1, num, dim]  能够控制一些重要信息进入后面的步骤
        user_summary = self.review_gate(user_word_embs_output)
        item_summary = self.review_gate(item_word_embs_output)
        user_feature = []
        item_feature = []

        for i in range(self.head):
            r_coatt = self.summary_coatt[i]
            w_coatt = self.word_coatt[i]

            # ------------------summary-level co-attention ---------------------------------
            # [1, user_num, 1], [1, item_num, 1]   根据summary选出了每一个num中比较重要的词
            user_pointers, item_pointers = r_coatt(user_summary, item_summary)

            # ------------------word-level co-attention ---------------------------------
            # [1, dim, num] @ [1, num, 1] = [1, dim, 1]  提取对于每一个user/item的比较重要内容
            user_sum_words = user_all_summary.transpose(1, 2).float() @ user_pointers
            item_sum_words = item_all_summary.transpose(1, 2).float() @ item_pointers

            # # [1, dim, 1]
            # user_sum_words = user_sum_words.squeeze(2).long()
            # item_sum_words = item_sum_words.squeeze(2).long()

            user_words = self.user_summary_embs(user_sum_words.squeeze(2).long())
            item_words = self.item_summary_embs(item_sum_words.squeeze(2).long())
            user_pointers, item_pointers = w_coatt(user_words, item_words)
            uset_words_feature = user_words.permute(0, 2, 1).bmm(user_pointers).squeeze(2)
            item_word_feature = user_words.permute(0, 2, 1).bmm(item_pointers).squeeze(2)
            user_feature.append(uset_words_feature)
            item_feature.append(item_word_feature)

        user_feature = torch.cat(user_feature, dim=1)
        item_feature = torch.cat(item_feature, dim=1)

        user_feature = self.drop_out(self.fc_user(user_feature))
        item_feature = self.drop_out(self.fc_item(item_feature))

        return torch.stack([user_feature], dim=1), torch.stack([item_feature], dim=1)

    def review_gate(self, reviews):
        reviews = reviews.sum(2)
        return torch.sigmoid(self.fc_gate1(reviews)) * torch.tanh(self.fc_gate2(reviews))

    def init_params(self):
        for fc in [self.fc_gate1, self.fc_gate2, self.fc_user[0], self.fc_user[-1], self.fc_item[0], self.fc_item[-1]]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)

        # if self.config.use_word_embedding:
        #     w2v = torch.from_numpy(np.load(self.config.w2v_path))
        #     if self.config.use_gpu:
        #         self.user_word_embs.weight.data.copy_(w2v.cuda())
        #         self.item_word_embs.weight.data.copy_(w2v.cuda())
        #     else:
        #         self.user_word_embs.weight.data.copy_(w2v)
        #         self.item_word_embs.weight.data.copy_(w2v)
        # else:
            nn.init.uniform_(self.user_summary_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_summary_embs.weight, -0.1, 0.1)


class Co_Attention(nn.Module):
    '''
    review-level and word-level co-attention module
    Eq (2,3, 10,11)
    '''
    def __init__(self, dim, gumbel, pooling):
        super(Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_user = nn.Linear(dim, dim)
        self.fc_item = nn.Linear(dim, dim)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.M, gain=1)
        nn.init.uniform_(self.fc_user.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_user.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_item.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_item.bias, -0.1, 0.1)

    def forward(self, user_feature, item_feature):
        """
        :param user_feature:
        :param item_feature:
        :return:
        """
        # [1, user_num, dim]
        fc_user_output = self.fc_user(user_feature)
        # [1, item_num, dim]
        fc_item_output = self.fc_item(item_feature)

        # [1, user_num, dim]
        S = fc_user_output @ self.M
        # [1, user_num, item_num]
        S = S @ fc_item_output.transpose(1, 2)

        # user_score: [1, user_num] 把item的部分max/mean掉,得到每一条user(评论)的atten_score
        # item_score: [1, item_num] 把user的部分max/mean掉, 得到每一个item的atten_score
        if self.pooling == 'max':
            user_score = S.max(dim=2)[0]
            item_score = S.max(dim=1)[0]
        else:
            user_score = S.mean(dim=2)
            item_score = S.mean(dim=1)

        # user_score: [1, user_num] 得到每一条user(评论)的atten_weight
        # item_score: [1, item_num] 得到每一个item的atten_weight
        if self.gumbel:
            user_pointers = F.gumbel_softmax(user_score, hard=True, dim=1)
            item_pointers = F.gumbel_softmax(item_score, hard=True, dim=1)
        else:
            user_pointers = F.softmax(user_score, dim=1)
            item_pointers = F.softmax(item_score, dim=1)

        # [1, user_num, 1], [1, item_num, 1]
        return user_pointers.unsqueeze(2), item_pointers.unsqueeze(2)