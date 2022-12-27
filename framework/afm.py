import torch.nn as nn
import torch
import torch.nn.functional as F


class AFM(nn.Module):
    def __init__(self, config):
        super(AFM, self).__init__()
        self.config = config
        self.dim = config.after_fusion_dim
        self.k = 10

        self.fc = nn.Linear(config.after_fusion_dim, self.k )

        self.fm_V = nn.Parameter(torch.randn(config.after_fusion_dim, self.k ))
        self.b_users = nn.Parameter(torch.randn(config.user_num, 1))
        self.b_items = nn.Parameter(torch.randn(config.item_num, 1))

        self.attenLayer = AttentionLayer(self.k)

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, -0.05, 0.05)
        nn.init.constant_(self.fc.bias, 0.0)
        # nn.init.uniform_(self.b_users, a=0, b=0.1)
        # nn.init.uniform_(self.b_items, a=0, b=0.1)
        nn.init.uniform_(self.fm_V, -0.05, 0.05)

    def fm_layer(self, input_vec):
        """
        b = 1
        :param input_vec: [1, fusion_dim]
        :return:
        """
        # [1, k]
        fm_linear = self.fc(input_vec)

        # fm_interactions: [1, k]
        fm_interactions_1 = (input_vec @ self.fm_V) ** 2
        fm_interactions_2 = input_vec ** 2 @ self.fm_V ** 2

        #  fm = b = XW + 0.5 * sum( (XV)∘(XV) − (X∘X)(V∘V) )
        fm_interaction = fm_linear + fm_interactions_1 * fm_interactions_2
        atten_interaction = self.attenLayer(fm_interaction)
        return atten_interaction

    def forward(self, feature, data):
        # user_id, item_id, _, _, _, _, _ = data
        # user_id, item_id = user_id.to(self.config.device), item_id.to(self.config.device)
        fm_out = self.fm_layer(feature)
        return fm_out


class AttentionLayer(nn.Module):
    def __init__(self, k):
        super(AttentionLayer, self).__init__()
        self.atten_weight = nn.Linear(k, k)
        self.atten_dense = nn.Linear(k, k)

        self.init_weight()
    def init_weight(self):
        nn.init.uniform_(self.atten_weight.weight, -0.05, 0.05)
        nn.init.constant_(self.atten_weight.bias, 0.0)
        nn.init.uniform_(self.atten_dense.weight, -0.05, 0.05)
        nn.init.constant_(self.atten_dense.bias, 0.0)

    def forward(self, interation):
        """
        b = 1
        :param interation: [b, k]
        :return:
        """
        # atten: [1, k]
        atten = self.atten_weight(interation)
        atten = F.relu(atten)
        # atten_scores: [1, k]
        atten_scores = self.atten_dense(atten)
        # atten_weight: [1, k]
        atten_weight = F.softmax(atten_scores, dim=1)

        atten_out = torch.sum(atten_weight * interation, dim=1).unsqueeze(1)
        return atten_out