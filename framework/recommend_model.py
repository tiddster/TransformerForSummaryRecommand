
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionLayer(nn.Module):
    '''
        Rating Prediciton Methods
        - LFM: Latent Factor Model
        - (N)FM: (Neural) Factorization Machine
        - MLP
        - SUM
    '''
    def __init__(self, config):
        super(PredictionLayer, self).__init__()
        self.output_type = 'lfm'

        if config.predictionLayerOutputType == "fm":
            self.model = FM(config)
        elif config.predictionLayerOutputType == "lfm":
            self.model = LFM(config)
        elif config.predictionLayerOutputType == 'mlp':
            self.model = MLP(config)
        elif config.predictionLayerOutputType == 'nfm':
            self.model = NFM(config)

    def forward(self, feature, data):
        user_id, item_id, _, _, _, _, _ = data

        if self.output_type == "lfm" or "fm":
            return self.model(feature, user_id, item_id)
        else:
            return self.model(feature)


class LFM(nn.Module):

    def __init__(self, config):
        super(LFM, self).__init__()

        self.fc = nn.Linear(config.feature_dim, 1)

        self.b_users = nn.Parameter(torch.randn(config.user_num, 1))
        self.b_items = nn.Parameter(torch.randn(config.item_num, 1))

        self.init_params()

    def init_params(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc.bias, a=0.5, b=1.5)
        nn.init.uniform_(self.b_users, a=0.5, b=1.5)
        nn.init.uniform_(self.b_items, a=0.5, b=1.5)

    def rescale_sigmoid(self, score, a, b):
        return a + torch.sigmoid(score) * (b - a)

    def forward(self, feature, user_id, item_id):
        # return self.rescale_sigmoid(self.fc(feature), 1.0, 5.0) + self.b_users[user_id] + self.b_items[item_id]
        fc_output = self.fc(feature)
        return fc_output + self.b_users[user_id] + self.b_items[item_id]

# ???
class NFM(nn.Module):
    '''
    Neural FM
    '''
    def __init__(self, config):
        super(NFM, self).__init__()
        self.dim = config.after_fusion_dim

        self.fc = nn.Linear(self.dim, 1)

        self.fm_V = nn.Parameter(torch.randn(16, self.dim))
        self.mlp = nn.Linear(16, 16)
        self.h = nn.Linear(16, 1, bias=False)
        self.drop_out = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)
        nn.init.uniform_(self.fm_V, -0.1, 0.1)
        nn.init.uniform_(self.h.weight, -0.1, 0.1)

    def forward(self, feature):
        fm_linear_part = self.fc(feature)
        fm_interactions_1 = torch.mm(feature, self.fm_V.t())
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(feature, 2), torch.pow(self.fm_V, 2).t())
        bilinear = 0.5 * (fm_interactions_1 - fm_interactions_2)

        out = F.relu(self.mlp(bilinear))
        out = self.drop_out(out)
        out = self.h(out) + fm_linear_part
        return out


class FM(nn.Module):

    def __init__(self, config):
        super(FM, self).__init__()
        self.dim = config.feature_dim

        self.fc = nn.Linear(config.feature_dim, 1)

        self.fm_V = nn.Parameter(torch.randn(config.feature_dim, 10))
        self.b_users = nn.Parameter(torch.randn(config.user_num, 1))
        self.b_items = nn.Parameter(torch.randn(config.item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def build_fm(self, input_vec):
        '''
        y = w_0 + \sum {w_ix_i} + \sum_{i=1}\sum_{j=i+1}<v_i, v_j>x_ix_j
        factorization machine layer
        refer: https://github.com/vanzytay/KDD2018_MPCN/blob/master/tylib/lib
                      /compose_op.py#L13
        '''
        # linear part: first two items
        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = input_vec @ self.fm_V
        fm_interactions_1 = fm_interactions_1 ** 2

        fm_interactions_2 = input_vec ** 2 @ self.fm_V ** 2
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2, dim=1, keepdim=True) + fm_linear_part
        return fm_output

    def forward(self, feature, uids, iids):
        fm_out = self.build_fm(feature)
        return fm_out + self.b_users[uids] + self.b_items[iids]


class MLP(nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.dim = config.feature_dim

        self.fc = nn.Linear(config.feature_dim, 1)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, 0.1, 0.1)
        nn.init.uniform_(self.fc.bias, a=0, b=0.2)

    def forward(self, feature):
        output_fc = self.fc(feature)

        return F.relu(output_fc)