import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    '''
    混合层：将user_feature和item_feature关联起来
    '''
    def __init__(self, config):
        super(FusionLayer, self).__init__()
        self.attn = SelfAtt(config.id_emb_dim, config.num_heads)

        self.config = config

        self.linear = nn.Linear(config.feature_dim, config.feature_dim)
        self.drop_out = nn.Dropout(0.5)

        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, user_feature, item_feature):
        """
        b=1
        :param user_feature: [b, 2, dim]
        :param item_feature: [b, 2, dim]
        :return:
        """
        output = self.attn(user_feature, item_feature)

        # att_user_out: [1, 2, 64]
        # att_item_out: [1, 2, 64]
        att_user_out, att_item_out = torch.split(output, output.size(1) // 2, dim=1)

        user_feature = user_feature + att_user_out
        item_feature = item_feature + att_item_out

        # [1, 2 * dim]
        user_feature = user_feature.reshape(user_feature.size(0), -1)
        item_feature = item_feature.reshape(item_feature.size(0), -1)
        # if self.config.r_id_merge == 'cat':
        #     [1, 2 * dim]
        #     user_feature = user_feature.reshape(user_feature.size(0), -1)
        #     item_feature = item_feature.reshape(item_feature.size(0), -1)
        # else:
        #   [1, dim]
        #    user_feature = user_feature.sum(dim=1)
        #    item_feature = item_feature.sum(dim=1)

        # output: [1, 2 * 2 * dim]
        output = torch.cat([user_feature, item_feature], dim=1)
        # if self.config.ui_merge == 'cat':
        #     out = torch.cat([user_feature, item_feature], dim=1)
        # elif self.config.ui_merge == 'add':
        #     out = user_feature + item_feature
        # else:
        #     out = user_feature * item_feature

        # out = self.drop_out(out)
        # return F.relu(self.linear(out))
        return output


class SelfAtt(nn.Module):
    '''
    self attention for interaction
    '''
    def __init__(self, dim, num_heads):
        super(SelfAtt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 2)

    def forward(self, user_feature, item_feature):
        """
        b = 1
        :param user_feature:  [b, 2, dim]
        :param item_feature:  [b, 2, dim]
        :return:
        """

        # feature: [1, 2, dim]
        feature = torch.cat([user_feature, item_feature], dim=1).permute(1, 0, 2)
        output = self.encoder(feature)

        # output: [1, 4, dim]
        output = output.permute(1, 0, 2)
        return output