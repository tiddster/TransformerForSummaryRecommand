import time

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm

import dataset.preprocess as pre
import feature_models.narre as narre
import feature_models.transSummary as trans
from model import Model
from feature_models import mpcn, tarmf, transSummary

import json


def train(train_info_data=None):
    # 训练模式更改：利用过拟合判断是否暂停训练
    if train_info_data is None:
        num_epoch = 0
        loss_up_num = 0  # loss连续上升的次数
        train_loss_list, min_test_loss = [9999], 9999
    else:
        num_epoch = train_info_data["epoch"]
        train_loss_list = train_info_data["train_loss_list"]
        min_test_loss = train_info_data["min_test_loss"]
        loss_up_num = train_info_data["loss_up_num"]

    # for epoch in range(num_epoch):
    while loss_up_num <= 3:
        is_model_save = True    # 根据train_loss是否下降判断是否保存模型
        is_train_stop = False  # 若两次loss小于1e-5则提前结束训练
        num_epoch += 1
        result = f"epoch: {num_epoch}  ====>  "
        start = time.time()
        # ---------------------------训练模式--------------------------------
        train_total_loss, train_total_num = 0.0, 0
        model.train()
        for data in tqdm(train_iter):
            _, _, _, _, rating, _, _ = data
            rating = torch.tensor(rating, dtype=torch.float).to(config.device)
            output = model(data)

            if lossType == 'rmse':
                loss = mse_criterion(output, rating)
                loss = torch.sqrt(loss)
            elif lossType == 'mae':
                loss = mae_criterion(output, rating)
            else:
                loss = mse_criterion(output, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_total_num += rating.shape[0]

        train_loss = train_total_loss / train_total_num
        result += f"train_loss: {train_loss}  "

        # 如果两次loss相差小于1e-6, 则提前结束
        if abs(train_loss - train_loss_list[-1]) < 1e-5:
            is_train_stop = True

        # 如果train_loss比上次的大，则不保存模型，loss_up_num+1
        if train_loss > train_loss_list[-1]:
            loss_up_num += 1
            is_model_save = False
        else:
            train_loss_list.append(train_loss)
            is_model_save = True

        # ---------------------------验证模式--------------------------------
        # model.eval()
        # val_total_loss, val_total_num = 0.0, 0
        # for data in tqdm(val_iter):
        #     user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary = data
        #     rating = torch.tensor(rating, dtype=torch.float).to(config.device)
        #     output = model(data)
        #
        #     loss = mse_criterion(output, rating)
        #     if lossType == 'rmse':
        #         loss = torch.sqrt(loss)
        #
        #     val_total_loss += loss.item()
        #     val_total_num += rating.shape[0]
        #
        # avg_loss = val_total_loss / val_total_num
        # result += f"val_loss: {avg_loss}  "
        # val_loss.append(avg_loss)

        # ---------------------------测试模式--------------------------------
        model.eval()
        test_total_num, test_total_acc, test_total_loss = 0, 0, 0.0
        # rating_list, output_list = [], []
        for data in tqdm(test_iter):
            user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary = data
            rating = rating.to(config.device)
            output = model(data)
            # output = torch.tensor(output, dtype=torch.int).to(config.device)

            rating_item = rating.item()
            output_item = output.item()

            if lossType == 'rmse':
                loss = mse_criterion(output, rating)
                loss = torch.sqrt(loss)
            elif lossType == 'mae':
                loss = mae_criterion(output, rating)
            else:
                loss = mse_criterion(output, rating)

            test_total_loss += loss.item()
            # abs_diff = abs(rating_item - output_item)
            # test_total_acc += 1 if abs_diff <= 0.5 else 0
            test_total_num += rating.shape[0]

            # rating_list.append(rating_item)
            # output_list.append(output_item)

        # test_acc = test_total_acc / test_total_num
        test_loss = test_total_loss / test_total_num
        result += f"test_loss:{test_loss}"
        # ---------------------------最后处理--------------------------------
        end = time.time()
        result += f" time: {end - start}"
        print(result)

        # 如果出现了nan则是过拟合，则直接提前结束，不保存模型
        if torch.any(torch.isnan(loss)):
            print("过拟合")
            break

        # 如果test_loss的最小值更新了，则保存模型
        min_test_loss = min(test_loss, min_test_loss)
        if min_test_loss == test_loss:
            is_model_save = True

        if is_model_save:
            train_info_data = {"epoch": num_epoch, "train_loss_list": train_loss_list, "min_test_loss": min_test_loss,
                               "loss_up_num": loss_up_num}
            save_model(model, train_info_data, model_name)
            print("储存模型......")

        if is_train_stop:
            break


def loss_plot(train_loss, val_loss=None, epoch_num=0):
    x = range(0, epoch_num, 1)
    # plt.figure()
    plt.plot(x, train_loss)
    # plt.plot(x, val_loss)
    # plt.xlabel('feature_num')
    # plt.ylabel('test_accuracy')
    # plt.legend()
    plt.show()


def save_model(model, train_info_data, filename):
    # 保存模型
    model_path = f"P:\\TransformerAFM_MODEL\\{filename}.pt"
    torch.save(model.state_dict(), model_path)

    # 保存训练loss、最小test_loss、loss增加次数
    json_path = f"save_final_model\\{filename}.json"
    info_json = json.dumps(train_info_data, sort_keys=False, indent=4, separators=(',', ': '))
    f = open(json_path, 'w')
    f.write(info_json)
    f.close()


def load_model(model, filename):
    # 读取模型
    model_path = f"P:\\TransformerAFM_MODEL\\{filename}.pt"
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'该model参数总量: {total_num}   可训练参数总量: {trainable_num}')

    # 读取json文件模型相关数据
    json_path = f"save_final_model\\{filename}.json"
    f = open(json_path, 'r')
    train_info_data = json.load(f)

    return model, train_info_data


if __name__ == '__main__':
    # num_epoch = 0
    train_iter, test_iter, config = pre.get_dataiter()
    model_name = f"Transformer_AFM_{config.feature_dim}_{config.num_heads}"
    print(model_name)

    narreM = narre.NARRE
    mpcnM = mpcn.MPCN
    tarmfM = tarmf.TARMF
    transM = trans.TRANSFORMER

    model = Model(config, transM).to(config.device)
    train_info_data = None
    # model, train_info_data = load_model(model, model_name)
    lossType = config.lossType

    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train(train_info_data)
    # loss_plot(train_loss, num_epoch)

# # -------------------------------测试模型------------------------------------
# if __name__ == '__main__':
#     train_iter, test_iter, val_iter, config = pre.get_dataiter()
#     transM = tfui.TRANSFORMER(config).to(config.device)
#
#     for data in train_iter:
#         user_feature, item_feature = transM(data)
