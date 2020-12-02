import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as fun
from .base_model_v3 import Model
from tensorboardX import SummaryWriter


# 没有想到好的名字就用contrastive recommendation来代表这个model吧
class CR(Model):
    """"""
    def __init__(self, args, dataset, filename):
        super(CR, self).__init__()
        self.args = args
        self.batch_size = self.args.bsz
        self.filename = filename
        self.data_list = dataset
        self.user_size = len(self.data_list.user_set)
        self.item_size = len(self.data_list.item_set)
        self.sz = self.data_list.train_list.shape[0]

        # user id embedding
        self.user_matrix = nn.Embedding(self.user_size, self.args.dim)
        self.user_matrix = self.user_matrix.cuda()
        self.user_matrix = nn.init.normal_(self.user_matrix.weight, std=0.01)

        # item id embedding
        self.item_matrix = nn.Embedding(self.item_size, self.args.dim)
        self.item_matrix = self.item_matrix.cuda()
        self.item_matrix = nn.init.normal_(self.item_matrix.weight, std=0.01)

        # item feature vectors 需要对这个加一层mlp 和 relu函数，使输出后的结果尽可能靠近embedding
        # positive sample 是 item_matrix里面
        self.item_features = torch.from_numpy(self.data_list.item_set)
        self.item_features.requires_grad = False
        self.item_features = self.item_features.cuda()
        dim_mlp = self.item_features.shape[1]
        self.item_features = self.item_features.float()
        self.item_mlp = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.LeakyReLU(), nn.Linear(dim_mlp, dim_mlp))  #
        self.item_mlp = self.item_mlp.cuda()

    def predict(self, uid, iid):
        """
        uid of user_matrix
        iid of item_matrix
        :return:
        """
        p1 = self.user_matrix[uid]
        p2 = self.item_matrix[iid]
        return torch.sum(p1 * p2, dim=1)

    def predict_cold(self, uid, iid):
        """
        uid of user_matrix
        iid of item_matrix
        :return:
        """
        p1 = self.user_matrix[uid]
        item_fixed = self.item_mlp(self.item_features)
        p2 = item_fixed[iid]
        return torch.sum(p1 * p2, dim=1)

    def forward(self, item_fixed, user_ids, item_ids):
        user_ids_unique = torch.unique(user_ids)
        item_ids_unique = torch.unique(item_ids)

        pos_item_embedding = self.item_matrix[item_ids]
        all_item_embedding = self.item_matrix[item_ids_unique]

        pos_user_embedding = self.user_matrix[user_ids]
        all_user_embedding = self.user_matrix[user_ids_unique]

        pos_feature = item_fixed[item_ids]
        all_feature = item_fixed[item_ids_unique]

        contrastive_loss_1 = self.contrastive_loss(pos_item_embedding, pos_feature, all_feature) * self.args.con_weight
        contrastive_loss_2 = self.contrastive_loss(pos_feature, pos_user_embedding, all_user_embedding)
        contrastive_loss_3 = self.contrastive_loss(pos_item_embedding, pos_user_embedding, all_user_embedding)

        reg_loss = ((all_item_embedding ** 2).mean() + (all_user_embedding ** 2).mean() + (all_feature ** 2).mean())
        reg_loss = reg_loss / 3
        return (contrastive_loss_1 + contrastive_loss_2 + contrastive_loss_3) / 3, reg_loss

    def contrastive_loss(self, tensor_anchor, tensor_pos, tensor_ll):
        """"""
        pos_score = torch.sum(tensor_anchor * tensor_pos, dim=1)
        all_score = torch.matmul(tensor_anchor, tensor_ll.t())
        pos_score_exp = torch.exp(pos_score / self.args.temp_value)
        all_score_exp = torch.exp(all_score / self.args.temp_value)
        all_score_exp_sum = torch.sum(all_score_exp, dim=1)
        contrastive_loss = (-torch.log(pos_score_exp / all_score_exp_sum)).mean()
        return contrastive_loss

    def final_loss(self, item_fixed, user_ids, item_ids):

        contrastive_loss, reg_loss = self.forward(item_fixed, user_ids, item_ids)
        reg_loss = self.args.reg_weight * reg_loss
        return contrastive_loss + reg_loss

    def regs(self, uid, iid, jid):
        # regs:  default value is 0
        reg = self.args.reg
        uid_v = self.user_matrix[uid]
        iid_v = self.item_matrix[iid]
        jid_v = self.item_matrix[jid]
        emb_regs = torch.sum(uid_v * uid_v) + torch.sum(iid_v * iid_v) + torch.sum(jid_v * jid_v)
        return reg * emb_regs

    def train(self):

        print('cr is training')
        lr = self.args.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        epochs = self.args.epochs
        for epoch in range(epochs):

            generator = self.sample()  # 这里生成的
            while True:

                optimizer.zero_grad()
                item_fixed = self.item_mlp(self.item_features)

                s = next(generator)
                if s is None:
                    break

                uid, iid = s[:, 0], s[:, 1]
                uid = uid.cuda()
                iid = iid.cuda()
                # current_loop += 1
                loss = self.final_loss(item_fixed, uid, iid)
                loss.backward()
                optimizer.step()

            if epoch % 2 == 0 and epoch > 1:  #
                print(f'=={epoch}===>loss is {loss}======<')
                file_path = open(self.filename, 'a')
                print(f'epoch is {epoch}-----', file=file_path)
                self.val(), self.test(), self.test_warm(), self.test_cold()

    def sample(self):
        np.random.shuffle(self.data_list.train_list)
        loop_size = self.sz // self.batch_size
        for i in range(loop_size):
            pairs = []
            sub_train_list = self.data_list.train_list[i * self.batch_size:(i + 1) * self.batch_size, :]
            for m, j in sub_train_list:
                pairs.append((m, j))

            yield torch.LongTensor(pairs)
        yield None
