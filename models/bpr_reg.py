"""BPR with an explicit L2 regularization term.

RecBole's built-in BPR (recbole/model/general_recommender/bpr.py) has no
`reg_lambda` parameter: its loss is the plain pairwise BPR loss. This class
copies that model and adds

    reg_lambda * (||u||^2 + ||i+||^2 + ||i-||^2)

to the loss, so the regularization strength can be swept for the homework.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class BPRReg(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_size = config["embedding_size"]
        self.reg_lambda = config["reg_lambda"]

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()  # sum of squared L2 norms, averaged over batch

        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        return self.user_embedding(user), self.item_embedding(item)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)

        loss = self.bpr_loss(pos_score, neg_score)
        # require_pow=True -> (1/2) * sum ||e||^2 / batch_size, the usual L2 term
        reg = self.reg_loss(user_e, pos_e, neg_e, require_pow=True)
        return loss + self.reg_lambda * reg

    def predict(self, interaction):
        user_e, item_e = self.forward(interaction[self.USER_ID], interaction[self.ITEM_ID])
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user_e = self.user_embedding(interaction[self.USER_ID])
        return torch.matmul(user_e, self.item_embedding.weight.t()).view(-1)
