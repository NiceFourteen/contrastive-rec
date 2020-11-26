import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score


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
        boundary = torch.mean(pre_ds)
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
            'auc':  self.auc_calculate(gt, preds),
            'ndcg': Metric.ndcg(gt, preds)

        }
        return ret

    def __logscore(self, scores):
        metrics = list(scores.keys())
        metrics.sort()
        file = open('results.log', 'a')
        print(' '.join(['%s: %s' % (m, str(scores[m])) for m in metrics]))  # , file=file
        # self.logging.info(' '.join(['%s: %s' % (m,str(scores[m])) for m in metrics]))

    def test(self):
        print('----- test -----begin-----')
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
        print('----- val -----begin-----')
        u = torch.LongTensor(range(self.user_size))
        results = self.compute_results(u, self.data_list.val_samples)
        scores = self.compute_scores(self.data_list.val_gt, results)
        self.__logscore(scores)
        print('----- val -----end-----')

    def test_warm(self):
        print('----- test_warm -----begin-----')
        u = self.data_list.test_warm_u
        u = u.cuda()
        results = self.compute_results(u, self.data_list.test_warm_samples)
        scores = self.compute_scores(self.data_list.test_warm_gt, results)
        self.__logscore(scores)
        print('----- test_warm -----end-----')

    def test_cold(self):
        print('----- test_cold -----begin-----')
        u = self.data_list.test_cold_u
        u = u.cuda()
        results = self.compute_results_cold(u, self.data_list.test_cold_samples)
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
        gt = gt.cuda()
        preds = torch.from_numpy(preds)
        preds = preds.cuda()
        return roc_auc_score(gt, preds, average='samples')  # 两个nd-array 进行auc计算