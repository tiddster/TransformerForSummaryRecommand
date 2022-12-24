import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

class F1SCORE(nn.Module):
    def __init__(self):
        super(F1SCORE, self).__init__()

    def forward(self, labels, preds):
        """
        类型都是python list
        :param labels:
        :param preds:
        :return:
        """
        F1_score = f1_score(labels, preds)
        AUC_score = roc_auc_score(labels, preds)
        return F1_score, AUC_score
