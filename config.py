class Config():
    def __init__(self):
        self.self_att = True  # 是否在混合层中使用自注意力层
        self.num_heads = 6  # 多头注意力的头数

        self.vocab_size = 30000  # 词典大小
        self.feature_dim = 128  # 特征的维度
        self.summary_dim = 128
        self.id_emb_size = 64  # id嵌入层维度
        self.filters_num = 32

        self.use_word_embedding = False

        self.user_num = 0  # 用户的数量
        self.item_num = 0  # 评论数量

        self.drop_out = 0.5

        self.data_list = None