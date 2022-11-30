import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

PATH = "Automotive_5.json"
BERT_PATH = "P:\Dataset\Bert-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

class PreProcess():
    def __init__(self):
        self.data_list = self.read_json()

    # ---------------------------------------------------------------------------
    # 读取AutoMotive的数据, 输出summary和用户id
    def read_json(self):
        with open(PATH, 'r', encoding='utf-8') as fp:
            data_list = []
            for line in fp.readlines():
                json_data = json.loads(line)
                data_list.append(json_data)
        return data_list

    #---------------------------------------------------------------------------
    # 将用户id, 商品id整理成set_list和dict
    def build_id_info(self):
        data_list = self.data_list
        item_set = []
        user_set = []
        for data in data_list:
            user_set.append(data["reviewerID"])
            item_set.append(data["asin"])

        item_set = list(set(item_set))
        user_set = list(set(user_set))
        item_dict = {item_id: i for i, item_id in enumerate(item_set)}
        user_dict = {user_id: i for i, user_id in enumerate(user_set)}
        return user_set, user_dict, item_set, item_dict

    # ---------------------------------------------------------------------------
    # 根据用户id，商品id整理相应的summary
    def build_specific_summary(self):
        data_list = self.data_list
        user_set, _, item_set, _ = self.build_id_info()
        user_sum_dict = {user_id:[] for user_id in user_set}
        item_sum_dict = {item_id:[] for item_id in item_set}

        for data in data_list:
            user_id = data["reviewerID"]
            summary = data["summary"]
            user_sum_dict[user_id].append(summary)

        for data in data_list:
            item_id = data["asin"]
            summary = data["summary"]
            item_sum_dict[item_id].append(summary)

        return user_sum_dict, item_sum_dict







