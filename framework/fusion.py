import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    '''
    混合层：将user_feature和item_feature关联起来
    '''
    def __init__(self, config):
        super(FusionLayer, self).__init__()
        if config.self_att:
            self.attn = SelfAtt(config.id_emb_size, config.num_heads)

        self.config = config
        self.linear = nn.Linear(config.feature_dim, config.feature_dim)
        self.drop_out = nn.Dropout(0.5)

        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, u_out, i_out):
        if self.config.self_att:
            out = self.attn(u_out, i_out)
            s_u_out, s_i_out = torch.split(out, out.size(1)//2, 1)
            u_out = u_out + s_u_out
            i_out = i_out + s_i_out

        if self.config.r_id_merge == 'cat':
            u_out = u_out.reshape(u_out.size(0), -1)
            i_out = i_out.reshape(i_out.size(0), -1)
        else:
            u_out = u_out.sum(1)
            i_out = i_out.sum(1)

        if self.config.ui_merge == 'cat':
            out = torch.cat([u_out, i_out], 1)
        elif self.config.ui_merge == 'add':
            out = u_out + i_out
        else:
            out = u_out * i_out

        # out = self.drop_out(out)
        # return F.relu(self.linear(out))
        return out


class SelfAtt(nn.Module):
    '''
    self attention for interaction
    '''
    def __init__(self, dim, num_heads):
        super(SelfAtt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, 128, 0.4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

    def forward(self, user_feature, item_feature):
        feature = torch.cat([user_feature, item_feature], 1).permute(1, 0, 2)  # batch * 6 * 64
        out = self.encoder(feature)
        return out.permute(1, 0, 2)