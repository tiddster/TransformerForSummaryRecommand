import json
import random

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

from config import Config

PATH = "P:\RecommandOnTransformer\dataset\Automotive_5.json"
BERT_PATH = "P:\Dataset\Bert-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)


class PreProcess():
    def __init__(self):

        self.user_dict = None
        self.item_dict = None
        self.user_set = None
        self.item_set = None

        self.train_data_list = None
        self.test_data_list = None
        self.val_data_list = None

        self.max_sum_len = None

        self.data_list = self.read_json()

    # ---------------------------------------------------------------------------
    # 读取AutoMotive的数据, 输出summary和用户id
    def read_json(self):
        print("-----读取数据-----")
        with open(PATH, 'r', encoding='utf-8') as fp:
            data_list = []
            for line in fp.readlines():
                json_data = json.loads(line)
                data_list.append(json_data)
        return data_list

    # -----------------------------------1---------------------------------------
    # 将summary转换为id并加上padding
    def build_summary_tokens_id(self):
        print("-----转换summary-----")
        data_list = self.data_list
        self.max_sum_len = 0
        for data in data_list:
            summary = data["summary"]
            sum_tokens = tokenizer.tokenize(summary)
            self.max_sum_len = max(self.max_sum_len, len(sum_tokens))

        for i in range(len(data_list)):
            summary = data_list[i]["summary"].lower()
            sum_tokens = ['[CLS]'] + tokenizer.tokenize(summary) + ['[SEP]']
            sum_tokens = sum_tokens + ['[PAD] '] * (self.max_sum_len + 2 - len(sum_tokens))
            sum_token_ids = tokenizer.convert_tokens_to_ids(sum_tokens)

            data_list[i]["summary_id"] = sum_token_ids

    # ------------------------------------2--------------------------------------
    # 将用户id, 商品id整理成set_list和dict, 并将所有id, rating整理成int形式, 更新data
    def build_id_info(self):
        print("-----建立id data-----")
        data_list = self.data_list

        item_set = []
        user_set = []
        for data in data_list:
            user_set.append(data["reviewerID"])
            item_set.append(data["asin"])

        self.item_set = list(set(item_set))
        self.user_set = list(set(user_set))
        self.item_dict = {item_id: i for i, item_id in enumerate(self.item_set)}
        self.user_dict = {user_id: i for i, user_id in enumerate(self.user_set)}

        for i in range(len(data_list)):
            item_id = data_list[i]["asin"]
            item_id = self.item_dict[item_id]
            user_id = data_list[i]["reviewerID"]
            user_id = self.user_dict[user_id]

            rating = int(data_list[i]["overall"])
            sum_token_ids = data_list[i]["summary_id"]
            # data_list[i]["user_id"] = user_id
            # data_list[i]["item_id"] = item_id
            # data_list[i]["rating"] = int(data_list[i]["overall"])

            # data_list 被更新成这样
            self.data_list[i] = {
                "user_id": user_id,
                "item_id": item_id,
                "rating": rating,
                "summary_ids": sum_token_ids
            }

    # ----------------------------------3----------------------------------------
    # 根据用户id，商品id整理相应的summary
    def build_specific_summary(self):
        print("-----检索用户和物品对应的评论-----")
        user_all_summary = {user_id: [] for user_id in self.user_dict.values()}
        item_all_summary = {item_id: [] for item_id in self.item_dict.values()}
        print("-----检索该用户评论过的所有物品-----")
        user2itemList = {user_id: [] for user_id in self.user_dict.values()}
        print("-----检索该物品被评论过的所有用户-----")
        item2userList = {item_id: [] for item_id in self.item_dict.values()}

        for i, data in enumerate(self.data_list):
            summary_ids = data['summary_ids']
            user_id = data['user_id']
            item_id = data['item_id']

            user_all_summary[user_id].append(summary_ids)
            item_all_summary[item_id].append(summary_ids)

            user2itemList[user_id].append(item_id)
            item2userList[item_id].append(user_id)

        for i, data in enumerate(self.data_list):
            user_id = data['user_id']
            item_id = data['item_id']

            self.data_list[i]["user_all_summary"] = user_all_summary[user_id]
            self.data_list[i]["item_all_summary"] = item_all_summary[item_id]

            self.data_list[i]["user2itemList"] = user2itemList[user_id]
            self.data_list[i]["item2userList"] = item2userList[item_id]

    # ----------------------------------5---------------------------------------
    # 数据集不平衡，比例为 542 606 1430 3967  13928
    def balance_dataset(self):
        # 输出数据集的数量
        # one, two, three, four, five = 0, 0, 0, 0, 0
        new_data_list = []
        for data in self.data_list:
            rating = data["rating"]
            if rating == 1:
                for i in range(25):
                    new_data_list.append(data)
                # one += 1
            elif rating == 2:
                for i in range(21):
                    new_data_list.append(data)
                # two += 1
            elif rating == 3:
                for i in range(10):
                    new_data_list.append(data)
                # three += 1
            elif rating == 4:
                for i in range(4):
                    new_data_list.append(data)
                # four += 1
            elif rating == 5:
                new_data_list.append(data)
                # five += 1
        self.data_list = new_data_list
        # print(one, two, three, four, five)

    # ----------------------------------5---------------------------------------
    # 划分数据集
    def split_train_test(self):
        print("-----划分数据集-----")
        data_len = len(self.data_list)
        train_len = int(data_len * 0.6)
        test_len = int(data_len * 0.2)
        val_len = int(data_len * 0.2)

        random.shuffle(self.data_list)

        self.train_data_list = self.data_list[:train_len]
        self.test_data_list = self.data_list[train_len:train_len + test_len]
        self.val_data_list = self.data_list[train_len + test_len:]

    # ----------------------------------5----------------------------------------
    # 总操作
    def main_preprocess(self):
        self.build_summary_tokens_id()
        self.build_id_info()
        self.build_specific_summary()
        # self.balance_dataset()
        self.split_train_test()


class AutomotiveDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        data = self.data_list[index]

        user_id = torch.tensor([data['user_id']]).long()
        item_id = torch.tensor([data['item_id']]).long()
        user2itemList = torch.tensor(data['user2itemList']).long()
        item2userList = torch.tensor(data['item2userList']).long()
        rating = torch.tensor([data['rating']]).long()
        user_all_summary = torch.tensor(data['user_all_summary']).long()
        item_all_summary = torch.tensor(data['item_all_summary']).long()

        return user_id, item_id, user2itemList, item2userList, rating, user_all_summary, item_all_summary

    def __len__(self):
        return len(self.data_list)


def get_dataiter():
    p = PreProcess()
    p.main_preprocess()

    config = init_config(p)

    train_dataset = AutomotiveDataset(p.train_data_list)
    test_dataset = AutomotiveDataset(p.test_data_list)
    val_dataset = AutomotiveDataset(p.val_data_list)

    train_iter = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_iter = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_iter, test_iter, val_iter, config


def init_config(p):
    config = Config()
    config.user_num = len(p.user_dict)+1
    config.item_num = len(p.item_dict)+1
    config.data_list = p.data_list
    config.BERT_PATH = BERT_PATH
    config.max_sum_len = p.max_sum_len

    return config