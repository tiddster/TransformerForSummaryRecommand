import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEmbeddings

class TARMF(nn.Module):
    def __init__(self, config):
        super(TARMF, self).__init__()
        self.config = config

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
        # self.summary_embedding = nn.Embedding(config.vocab_size, config.summary_dim)

        # bert_config = AutoConfig.from_pretrained(config.BERT_PATH)
        # self.summary_embedding2 = BertEmbeddings(bert_config)
        # print(self.summary_embedding2)

        self.u_i_id_embedding = nn.Embedding(ui_id_num, config.id_emb_dim)

        self.rnn_mha = RNN_MHAttention(config)

        self.review_linear = nn.Linear(config.feature_dim, config.id_emb_dim)
        self.id_linear = nn.Linear(config.id_emb_dim, config.id_emb_dim, bias=False)
        self.attention_linear = nn.Linear(config.id_emb_dim, 1)
        self.fc_layer = nn.Linear(config.feature_dim, config.id_emb_dim)

        self.dropout = nn.Dropout(config.drop_out)
        self.init_param()

    def init_param(self):
        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)

        # if self.config.use_word_embedding:
        #     w2v = torch.from_numpy(np.load(self.opt.w2v_path))
        #     if self.opt.use_gpu:
        #         self.word_embs.weight.data.copy_(w2v.cuda())
        #     else:
        #         self.word_embs.weight.data.copy_(w2v)
        # else:
        #     nn.init.xavier_normal_(self.summary_embedding.weight)

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
        # --------------- word embedding ----------------------------------
        # summary = self.summary_embedding(summary)
        # bs, num, seq_len, wd = summary.size()
        bs, num, seq_len = summary.size()

        # ids:  [1, dim]
        id_emb = self.id_embedding(ids).squeeze(1)

        # uiid_emb_output:  [1, seq_num, dim]
        uiid_emb_output = self.u_i_id_embedding(ids_list)

        # --------cnn for review--------------------

        # feature: [seq_num, seq_len, dim]
        feature = F.relu(self.rnn_mha(summary)).transpose(1, 2)

        # feature: [seq_num, dim]
        feature = F.max_pool1d(feature, feature.size(2)).squeeze(2)

        # feature: [1, seq_num, dim]
        feature = feature.view(-1, num, feature.size(1))

        # ------------------linear attention-------------------------------

        # review_linear_output: [1, seq_num, dim]
        review_linear_output = self.review_linear(feature)

        # id_linear_output: [1, seq_num, dim]
        id_linear_output = self.id_linear(F.relu(uiid_emb_output))

        # rs_mix: [1, seq_num, dim]
        rs_mix = F.relu(review_linear_output + id_linear_output)

        # att_score: [1, seq_num, 1]
        att_score = self.attention_linear(rs_mix)

        # att_score: [1, seq_num, 1]
        att_weight = F.softmax(att_score, 1)

        # summary_feature_output: [1, seq_num, filter_num]
        summary_feature = feature * att_weight

        # summary_feature_output: [1, filter_num]
        summary_feature = summary_feature.sum(1)
        summary_feature = self.dropout(summary_feature)
        # summary_feature_output: [1, dim]
        summary_feature_output = self.fc_layer(summary_feature)

        return torch.stack([id_emb, summary_feature_output], dim=1)


class RNN_MHAttention(nn.Module):
    def __init__(self, config):
        super(RNN_MHAttention, self).__init__()
        self.config = config

        self.summary_embedding = nn.Embedding(config.vocab_size, config.summary_dim)
        self.lstm = nn.LSTM(config.summary_dim, config.feature_dim, batch_first=True)
        self.mha = nn.MultiheadAttention(config.feature_dim, num_heads=config.num_heads, batch_first=True)

    def init_params(self):
        nn.init.xavier_normal_(self.summary_embedding.weight)
        nn.init.xavier_normal_(self.lstm.weight)

    def forward(self, summary):
        summary = summary.squeeze(0)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(summary.size()[-1]).bool().to(self.config.device)
        paddingMask = RNN_MHAttention.getPaddingMask(summary).to(self.config.device)

        summary_embs_output = self.summary_embedding(summary)

        lstm_output,(_, _) = self.lstm(summary_embs_output)
        mha_output, _ = self.mha(lstm_output, lstm_output, lstm_output, key_padding_mask=paddingMask, attn_mask=tgt_mask)

        return mha_output

    def getPaddingMask(tokens):
        """
        用于padding_mask
        """
        paddingMask = torch.zeros(tokens.size())
        paddingMask[tokens == 100] = -torch.inf
        return paddingMask