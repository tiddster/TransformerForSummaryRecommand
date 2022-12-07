import dataset.preprocess as pre
import models.narre as narre


if __name__ == '__main__':
    train_iter, test_iter, val_iter, config = pre.get_dataiter()

    narreM = narre.NARRE(config)
    for data in train_iter:
        output = narreM(data)