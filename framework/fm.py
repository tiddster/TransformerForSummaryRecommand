import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, config):
        super(FM, self).__init__()
        self.config = config
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

    def forward(self, feature, data):
        user_id, item_id, _, _, _, _, _ = data
        user_id, item_id = user_id.to(self.config.device), item_id.to(self.config.device)
        fm_out = self.build_fm(feature)
        return fm_out + self.b_users[user_id] + self.b_items[item_id]