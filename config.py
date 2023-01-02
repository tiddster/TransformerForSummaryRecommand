import torch
class Config():
    def __init__(self):
        self.num_heads = 4  # 多头注意力的头数
        self.num_transformer_heads = 4

        # 这两个参数在模型实现的时候会重新定义
        self.num_feature = 2
        self.predictionLayerType = 'fm'

        self.vocab_size = 30522  # 词典大小
        self.feature_dim = 256 # 特征的维度
        self.summary_dim = 256
        self.id_emb_dim = 256  # id嵌入层维度
        self.after_fusion_dim = self.id_emb_dim * self.num_feature * 2
        self.kernel_size = 1
        self.filters_num = 20

        self.data_list = None
        self.user_num = 0  # 用户的数量
        self.item_num = 0  # 评论数量
        # self.max_sum_len = 0 # 最大summary长度
        # self.avg_sum_len = 0 # 平均summary长度

        self.self_att = True  # 是否在混合层中使用自注意力层
        self.use_word_embedding = False  # 是否使用预训练的词嵌入层

        self.lr = 1e-3
        self.weight_decay = 1e-3
        self.drop_out = 0.5

        self.r_id_merge = 'else'
        self.ui_merge = 'add'
        self.BERT_PATH = ""
        self.lossType = "rmse"  # "rmse" or "mae"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.get_device_name())