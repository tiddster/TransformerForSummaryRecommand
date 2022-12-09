import torch

import dataset.preprocess as pre
import models.narre as narre
from model import Model

if __name__ == '__main__':
    train_iter, test_iter, val_iter, config = pre.get_dataiter()

    narreM = narre.NARRE
    model = Model(config, narreM)

    for data in train_iter:
        output = model(data)
        print(output)

