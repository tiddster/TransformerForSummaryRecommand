from framework.afm import AFM
from framework.fusion import FusionLayer
from framework.recommend_model import PredictionLayer
from framework.fm import FM
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config, featuerLayer):
        super(Model, self).__init__()

        self.featureLayer = featuerLayer(config).to(config.device)
        self.fusionLayer = FusionLayer(config).to(config.device)

        self.config = config
        # 根据fusionLayer中的不同融合操作调整feature_dim = after_fusion_dim
        self.config.after_fusion_dim = 2 * config.num_feature * config.id_emb_dim

        self.predictionLayer = AFM(config).to(config.device)

        self.dropout = nn.Dropout(self.config.drop_out)

    def forward(self, data):
        user_feature, item_feature = self.featureLayer(data)
        fusion_feature = self.fusionLayer(user_feature, item_feature)
        fusion_feature = self.dropout(fusion_feature)
        output = self.predictionLayer(fusion_feature, data).squeeze(1)
        return output

