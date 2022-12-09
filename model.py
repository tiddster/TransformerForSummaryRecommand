from framework.fusion import FusionLayer
from framework.recommend_model import PredictionLayer

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config, featuerLayer):
        super(Model, self).__init__()

        self.config = config
        # 根据fusionLayer中的不同融合操作调整feature_dim
        self.config.feature_dim = 2 * 2 * config.id_emb_size

        self.featureLayer = featuerLayer(config)
        self.fusionLayer = FusionLayer(config)
        self.predictionLayer = PredictionLayer(config)

        self.dropout = nn.Dropout(self.config.drop_out)

    def forward(self, data):
        user_feature, item_feature = self.featureLayer(data)
        fusion_feature = self.fusionLayer(user_feature, item_feature)
        fusion_feature = self.dropout(fusion_feature)
        output = self.predictionLayer(fusion_feature, data).squeeze(1)
        return output

