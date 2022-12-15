import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm

import dataset.preprocess as pre
import models.narre as narre
import models.tfui as trans
from model import Model
from models import mpcn, tarmf, tfui

def train(num_epoch):
    train_loss, val_loss = [], []
    for epoch in range(num_epoch):
        result = ""
        start = time.time()
        # ---------------------------训练模式--------------------------------
        train_total_loss, train_total_num = 0.0, 0
        model.train()
        for data in tqdm(train_iter):
            user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary = data
            rating = torch.tensor(rating, dtype=torch.float).to(config.device)
            output = model(data)

            loss = mse_criterion(output, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_total_num += rating.shape[0]

        avg_loss = train_total_loss / train_total_num
        result += f"train_loss: {avg_loss}  "
        train_loss.append(avg_loss)

        # ---------------------------验证模式--------------------------------
        model.eval()
        val_total_loss, val_total_num = 0.0, 0
        for data in tqdm(val_iter):
            user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary = data
            rating = torch.tensor(rating, dtype=torch.float).to(config.device)
            output = model(data)

            loss = mse_criterion(output, rating)

            val_total_loss += loss.item()
            val_total_num += rating.shape[0]

        avg_loss = val_total_loss / val_total_num
        result += f"val_loss: {avg_loss}  "
        val_loss.append(avg_loss)

        # ---------------------------测试模式--------------------------------
        test_total_num, test_total_acc = 0, 0
        for data in tqdm(test_iter):
            user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary = data
            rating = rating.to(config.device)
            output = model(data)
            # output = torch.tensor(output, dtype=torch.int).to(config.device)

            abs_diff = torch.abs(rating-output).item()

            test_total_acc += 1 if abs_diff <= 0.5 else 0
            test_total_num += rating.shape[0]

        avg_acc = test_total_acc / test_total_num
        result += f"test_acc: {avg_acc * 100 :.2f}% "

        end = time.time()
        result += f"time: {end - start}"
        print(result)
    return train_loss, val_loss


def loss_plot(train_loss, val_loss, epoch_num):
    x = range(0, epoch_num, 1)
    # plt.figure()
    plt.plot(x, train_loss)
    plt.plot(x, val_loss)
    # plt.xlabel('feature_num')
    # plt.ylabel('test_accuracy')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    num_epoch = 10
    train_iter, test_iter, val_iter, config = pre.get_dataiter()

    # narreM = narre.NARRE
    # mpcnM = mpcn.MPCN
    tarmfM = tarmf.TARMF
    # transM = trans.TRANSFORMER
    model = Model(config, tarmfM).to(config.device)

    mse_criterion = nn.MSELoss()
    # mae_criterion = nn.L1Loss()
    # smooth_mae_criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_loss, val_loss = train(num_epoch)
    loss_plot(train_loss, val_loss, num_epoch)

# # -------------------------------测试模型------------------------------------
# if __name__ == '__main__':
#     train_iter, test_iter, val_iter, config = pre.get_dataiter()
#     transM = tfui.TRANSFORMER(config).to(config.device)
#
#     for data in train_iter:
#         user_feature, item_feature = transM(data)

