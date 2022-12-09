import torch

import dataset.preprocess as pre
import models.narre as narre
from framework.fusion import FusionLayer


if __name__ == '__main__':
    train_iter, test_iter, val_iter, config = pre.get_dataiter()

    narreM = narre.NARRE(config)
    fusionLayer = FusionLayer(config)

    for data in train_iter:
        user_feature, item_feature = narreM(data)
        fusion_feature = fusionLayer(user_feature, item_feature)

