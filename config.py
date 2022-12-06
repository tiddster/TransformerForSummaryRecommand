class Config():
    def __init__(self):
        self.self_att = True  # 是否在混合层中使用自注意力层
        self.num_heads = 6  # 多头注意力的头数

        self.feature_dim = 128 # 特征的维度

        self.id_emb_size = 128  # id嵌入层维度
