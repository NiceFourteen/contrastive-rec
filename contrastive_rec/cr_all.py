import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
import argparse
import os

parser = argparse.ArgumentParser(description='contrastive training for recommendation')
parser.add_argument('--ih', '--if_hard', default=False, type=bool,
                    metavar='IH', help='whether use hard negative samples', dest='ih')
parser.add_argument('--dim', '--dim_fea', default=64, type=int,
                    metavar='dim', help='the dim for item and user', dest='dim')
parser.add_argument('--lr', '--learning_rate_bpr', default=0.001, type=float,
                    metavar='LR', help='the learning rate for bpr training', dest='lr')
parser.add_argument('--reg', '--regular', default=0.01, type=float,
                    metavar='reg', help='regularization', dest='reg')
parser.add_argument('--con_weight', '--contra_weight', default=0.3, type=float,
                    metavar='con_weight', help='con_weight', dest='con_weight')
parser.add_argument('--reg_weight', '--reg_weight', default=0.5, type=float,
                    metavar='reg_weight', help='reg_weight', dest='reg_weight')
parser.add_argument('--temp_value', '--temp_value', default=2, type=float,
                    metavar='temp_value', help='temp_value', dest='temp_value')
parser.add_argument('--epochs', '--epoch_number', default=200, type=int,
                    metavar='epochs', help='training epochs', dest='epochs')
parser.add_argument('--dp', '--data_path', default='../../dataset/amazon/amazon/', type=str,
                    metavar='dp', help='the path of dataset', dest='dp')
parser.add_argument('--gpu', default=0, type=int,
                    metavar='GPU', help='gpu number to use', dest='gpu')
parser.add_argument('--bsz', default=10240, type=int,
                    metavar='bsz', help='batch_size', dest='bsz')


class Model(nn.Module):

    def compute_results(self, u, test_samples):
        u = u.cuda()
        if type(test_samples) != torch.Tensor:
            test_samples = torch.from_numpy(test_samples)
        test_samples = test_samples.cuda()
        rs = []
        for i in test_samples.T:
            # lt = torch.LongTensor(i)
            lt = i
            res_temp = self.predict(u, lt).detach()
            res_temp = res_temp.cpu()
            res_temp = res_temp.numpy()
            rs.append(res_temp)
        results = np.vstack(rs).T
        if np.isnan(results).any():
            raise Exception('nan')
        return results

    def compute_results_cold(self, u, test_samples):
        u = u.cuda()
        if type(test_samples) != torch.Tensor:
            test_samples = torch.from_numpy(test_samples)
        test_samples = test_samples.cuda()
        rs = []
        for i in test_samples.T:
            # lt = torch.LongTensor(i)
            lt = i
            res_temp = self.predict_cold(u, lt).detach()
            res_temp = res_temp.cpu()
            res_temp = res_temp.numpy()
            rs.append(res_temp)
        results = np.vstack(rs).T
        if np.isnan(results).any():
            raise Exception('nan')
        return results

    @staticmethod
    def auc_calculate(labels, pre_ds):
        print('.auc')
        labels = (torch.from_numpy(labels)).double()
        labels = labels.cuda()
        pre_ds = (torch.from_numpy(pre_ds)).double()
        pre_ds = pre_ds.cuda()
        # 计算tp，阈值设置为0.0，即预测集中，值大于0的样例设置为1，小于0的样例 设置为0
        one = torch.ones_like(pre_ds)
        zero = torch.zeros_like(pre_ds)
        boundary = torch.mean(pre_ds) * 0.3
        pre_ds = torch.where(pre_ds > boundary, one, zero)
        positive_lens = torch.sum(labels, dim=1)
        negative_lens = torch.abs(torch.sum(labels - 1, dim=1))
        positive_samples = torch.einsum('ij, ij->i', [labels, pre_ds])
        negative_samples = torch.einsum('ij, ij->i', [torch.abs(labels - 1), pre_ds])
        positive_results = positive_samples / positive_lens
        negative_results = negative_samples / negative_lens
        results = torch.mean((negative_results + positive_results) / 2.0)
        results = results.cpu()
        return np.array(results)

    def compute_scores(self, gt, preds):
        ret = {
            'auc':  Metric.auc(gt, preds),
            'ndcg': Metric.ndcg(gt, preds)

        }
        return ret

    def __logscore(self, scores):
        metrics = list(scores.keys())
        metrics.sort()
        file_path = open(self.filename, 'a')
        print(' '.join(['%s: %s' % (m, str(scores[m])) for m in metrics]), file=file_path)  #
        # self.logging.info(' '.join(['%s: %s' % (m,str(scores[m])) for m in metrics]))

    def test(self):
        file_path = open(self.filename, 'a')
        print('----- test', file=file_path)
        u = torch.LongTensor(range(self.user_size))
        u = u.cuda()
        test_arr = self.data_list.test_samples
        test_tensor = torch.from_numpy(test_arr)
        test_tensor = test_tensor.cuda()
        results = self.compute_results(u, test_tensor)
        scores = self.compute_scores(self.data_list.test_gt, results)
        self.__logscore(scores)
        print('----- test -----end-----')

    def val(self):
        file_path = open(self.filename, 'a')
        print('----- val', file=file_path)
        u = torch.LongTensor(range(self.user_size))
        results = self.compute_results(u, self.data_list.val_samples)
        scores = self.compute_scores(self.data_list.val_gt, results)
        self.__logscore(scores)
        print('----- val -----end-----')

    def test_warm(self):
        file_path = open(self.filename, 'a')
        print('----- test_warm', file=file_path)
        u = self.data_list.test_warm_u
        u = u.cuda()
        results = self.compute_results(u, self.data_list.test_warm_samples)
        scores = self.compute_scores(self.data_list.test_warm_gt, results)
        self.__logscore(scores)
        print('----- test_warm -----end-----')

    def test_cold(self):
        file_path = open(self.filename, 'a')
        print('----- test_cold', file=file_path)
        u = self.data_list.test_cold_u
        u = u.cuda()
        results = self.compute_results_cold(u, self.data_list.test_cold_samples)  # _cold
        scores = self.compute_scores(self.data_list.test_cold_gt, results)
        self.__logscore(scores)
        print('----- test_cold -----end-----')

    def train(self):
        raise Exception('no implementation')

    def regs(self):
        raise Exception('no implementation')

    def predict(self):
        raise Exception('no implementation')

    def predict_cold(self):
        raise Exception('no implementation')

    def save(self):
        raise Exception('no implementation')


class Metric:
    @staticmethod
    def get_annos(gt, preds):
        p_num = np.sum(gt > 0, axis=1, keepdims=True).flatten()
        # print(p_num)
        pos = np.argsort(-preds)[range(len(p_num)), p_num]
        # print(pos)
        ref_score = preds[range(len(pos)), pos]
        # print(preds.T, ref_score)
        annos = (preds.T - ref_score).T > 0
        return annos

    @staticmethod
    def ndcg(gt, preds):
        print('.ndcg')
        gt = torch.from_numpy(gt)
        preds = torch.from_numpy(preds)
        K = [5, 10, 20]
        return [ndcg_score(gt, preds, k=k) for k in K]  # 看这个地方具体怎么写的 修改这个地方的具体实现

    @staticmethod
    def auc(gt, preds):
        print('.auc')
        gt = torch.from_numpy(gt)
        # gt = gt.cuda()
        preds = torch.from_numpy(preds)
        # preds = preds.cuda()
        return roc_auc_score(gt, preds, average='samples')  # 两个nd-array 进行auc计算


class LoadData:
    def __init__(self, user_set,
                 item_set,
                 train_pt,
                 train_list,
                 test_cold_gt,
                 test_cold_samples,
                 test_cold_u,
                 test_samples,
                 test_warm_gt,
                 test_warm_samples,
                 test_warm_u,
                 val_cold_gt,
                 val_cold_samples,
                 val_cold_u,
                 val_warm_gt,
                 val_warm_samples,
                 val_warm_u,
                 val_samples,
                 val_gt,
                 test_gt):
        super(LoadData, self).__init__()
        self.user_set = user_set
        self.item_set = item_set
        self.train_pt = train_pt
        self.train_list = train_list
        self.test_cold_gt = test_cold_gt
        self.test_cold_samples = test_cold_samples
        self.test_cold_u = test_cold_u
        self.test_samples = test_samples
        self.test_warm_gt = test_warm_gt
        self.test_warm_samples = test_warm_samples
        self.test_warm_u = test_warm_u
        self.val_cold_gt = val_cold_gt
        self.val_cold_samples = val_cold_samples
        self.val_cold_u = val_cold_u
        self.val_warm_gt = val_warm_gt
        self.val_warm_samples = val_warm_samples
        self.val_warm_u = val_warm_u
        self.val_samples = val_samples
        self.val_gt = val_gt
        self.test_gt = test_gt


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
        # contrastive_loss_3 = self.contrastive_loss(pos_item_embedding, pos_user_embedding, all_user_embedding)

        reg_loss = (all_item_embedding ** 2).mean() + (all_user_embedding ** 2).mean() + (all_feature ** 2).mean()
        reg_loss = reg_loss / 3
        return (contrastive_loss_1 + contrastive_loss_2) / 2, reg_loss

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
                print(f'=={epoch}===>loss_bpr is {loss}======<')
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


"""
如果是随机选择negative samples，那么neg__num不能为None
"""


def main():
    args = parser.parse_args()
    lr_params = f'lr-{args.lr}-'
    con_weight_params = f'con_weight-{args.con_weight}-'
    reg_weight_params = f'reg_weight-{args.reg_weight}'
    temp_value_params = f'temp_value-{args.temp_value}'
    bsz_params = f'bsz-{args.bsz}-'
    filename = f'./result/cr-3-{lr_params}-{con_weight_params}-{reg_weight_params}-{temp_value_params}-{bsz_params}.txt'

    if not os.path.exists(filename):
        open(filename, 'x')

    if args.gpu is None:
        args.gpu = 0
    torch.cuda.set_device(args.gpu)

    base_path = args.dp
    # 读取预处理的数据
    user_set = torch.load(base_path + 'user_set.pt')
    item_set = torch.load(base_path + 'item_set.pt')
    train_pt = torch.load(base_path + 'train.pt')
    train_list = torch.load(base_path + 'train_list_as_array.pt')

    test_cold_gt = torch.load(base_path + 'test_cold_gt.pt')
    test_cold_samples = torch.load(base_path + 'test_cold_samples.pt')
    test_cold_u = torch.load(base_path + 'test_cold_u.pt')
    test_samples = torch.load(base_path + 'test_samples.pt')
    test_set = torch.load(base_path + 'test_set.pt')
    test_warm_gt = torch.load(base_path + 'test_warm_gt.pt')
    test_warm_samples = torch.load(base_path + 'test_warm_samples.pt')
    test_warm_u = torch.load(base_path + 'test_warm_u.pt')
    val_cold_gt = torch.load(base_path + 'val_cold_gt.pt')
    val_cold_samples = torch.load(base_path + 'val_cold_samples.pt')
    val_cold_u = torch.load(base_path + 'val_cold_u.pt')

    val_set = torch.load(base_path + 'val_set.pt')
    val_warm_gt = torch.load(base_path + 'val_warm_gt.pt')
    val_warm_samples = torch.load(base_path + 'val_warm_samples.pt')
    val_warm_u = torch.load(base_path + 'val_warm_u.pt')

    val_samples = torch.load(base_path + 'val_samples.pt')
    val_gt = torch.load(base_path + 'val_gt.pt')
    test_gt = torch.load(base_path + 'test_gt.pt')

    data = LoadData(user_set,
                    item_set,
                    train_pt,
                    train_list,
                    test_cold_gt,
                    test_cold_samples,
                    test_cold_u,
                    test_samples,
                    test_warm_gt,
                    test_warm_samples,
                    test_warm_u,
                    val_cold_gt,
                    val_cold_samples,
                    val_cold_u,
                    val_warm_gt,
                    val_warm_samples,
                    val_warm_u,
                    val_samples,
                    val_gt,
                    test_gt)
    print('data is ready')
    cr_model = CR(args, data, filename)
    cr_model = cr_model.cuda()
    cr_model.train()


if __name__ == '__main__':
    main()

