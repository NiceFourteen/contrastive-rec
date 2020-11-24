import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as fun
from .base_model import Model
import tqdm


# 没有想到好的名字就用contrastive recommendation来代表这个model吧
class CR(Model):
    """"""
    def __init__(self, args, dataset):
        super(CR, self).__init__()
        self.args = args
        self.batch_size = self.args.bsz
        self.data_list = dataset
        self.user_size = len(self.data_list.user_set)
        self.item_size = len(self.data_list.item_set)
        self.sz = self.data_list.train_list.shape[0]

        # user is embedding
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
        self.item_mlp = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.LeakyReLU())  #
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

    def bpr_loss(self, uid, iid, jid):
        """
        bpr的算法是，对一个用户u求i和j两个item的分数，然后比较更喜欢哪个，
        所以这里需要进行两次预测，分别是第i个item的和第j个item的
        """
        pre_i = self.predict(uid, iid)
        pre_j = self.predict(uid, jid)
        dev = pre_i - pre_j
        return torch.sum(fun.softplus(-dev))

    def con_loss(self, item_ids, item_fixed):
        """"""
        item_feature_v = item_fixed[item_ids]
        item_embedding_v = self.item_matrix[item_ids]
        loss = torch.sum(torch.einsum('ij, kj-> ik', [item_embedding_v, item_feature_v]))
        return loss

    def regs(self, uid, iid, jid):
        # regs:  default value is 0
        reg = 0
        uid_v = self.user_matrix[uid]
        iid_v = self.item_matrix[iid]
        jid_v = self.item_matrix[jid]
        emb_regs = torch.sum(uid_v * uid_v) + torch.sum(iid_v * iid_v) + torch.sum(jid_v * jid_v)
        return reg * emb_regs

    def train(self):
        """
        lr: learning rate default value is 0.01
        :return:
        """
        print('cr is training')
        lr_bpr = self.args.lr_bpr
        lr_con = self.args.lr_con
        wd_bpr = self.args.wd_bpr
        wd_con = self.args.wd_con
        optimizer_bpr = torch.optim.Adagrad([self.user_matrix, self.item_matrix],
                                            lr=lr_bpr, weight_decay=0)
        optimizer_con = torch.optim.SGD(self.item_mlp.parameters(), lr=lr_con, weight_decay=0)
        loop_size = self.sz // self.batch_size
        epochs = self.args.epochs
        for epoch in range(epochs):
            generator = self.sample()  # 这里生成的
            current_loop = 1
            while True:
                optimizer_bpr.zero_grad()
                optimizer_con.zero_grad()
                item_fixed = self.item_mlp(self.item_features)
                generator_tensors = next(generator)
                if generator_tensors is None:
                    break
                s = generator_tensors[0]
                sub_item_ids = generator_tensors[1]
                if s is None:
                    break
                if sub_item_ids is None:
                    break
                uid, iid, jid = s[:, 0], s[:, 1], s[:, 2]
                uid = uid.cuda()
                iid = iid.cuda()
                jid = jid.cuda()
                loss_bpr = self.bpr_loss(uid, iid, jid) + self.regs(uid, iid, jid)
                loss_con = self.con_loss(sub_item_ids, item_fixed)
                # print(f'=={epoch}=={current_loop}/{loop_size}=====>loss_bpr is {loss_bpr}=====loss_con is {loss_con}')
                current_loop += 1
                loss_bpr.backward()
                loss_con.backward()
                optimizer_bpr.step()
                optimizer_con.step()
            if epoch % 2 == 0:  # and epoch > 1

                print(f'epoch is {epoch}')
                self.val(), self.test(), self.test_warm(), self.test_cold()

    def sample(self):
        np.random.shuffle(self.data_list.train_list)
        shuffle_item_ids = torch.randperm(self.item_size)
        loop_size = self.sz // self.batch_size
        item_group_size = self.item_size // loop_size
        for i in range(loop_size):
            pairs = []
            sub_item_ids = shuffle_item_ids[i * item_group_size:(i + 1) * item_group_size]
            sub_train_list = self.data_list.train_list[i * self.batch_size:(i + 1) * self.batch_size, :]
            for m, j in sub_train_list:
                m_neg = j
                while m_neg in self.data_list.train_pt[m]:
                    m_neg = np.random.randint(self.item_size)
                pairs.append((m, j, m_neg))

            yield torch.LongTensor(pairs), sub_item_ids
        yield None
