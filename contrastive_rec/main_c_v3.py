import argparse
import torch
import os
from base_algorithm.load import LoadData
from contrastive_rec.cr_v3 import CR

parser = argparse.ArgumentParser(description='contrastive training for recommendation')
parser.add_argument('--ih', '--if_hard', default=False, type=bool,
                    metavar='IH', help='whether use hard negative samples', dest='ih')
parser.add_argument('--dim', '--dim_fea', default=64, type=int,
                    metavar='dim', help='the dim for item and user', dest='dim')
parser.add_argument('--lr', '--learning_rate_bpr', default=0.001, type=float,
                    metavar='LR', help='the learning rate for bpr training', dest='lr')
parser.add_argument('--reg', '--regular', default=0.01, type=float,
                    metavar='reg', help='regularization', dest='reg')
parser.add_argument('--con_weight', '--contra_weight', default=0.1, type=float,
                    metavar='con_weight', help='con_weight', dest='con_weight')
parser.add_argument('--reg_weight', '--reg_weight', default=0.7, type=float,
                    metavar='reg_weight', help='reg_weight', dest='reg_weight')
parser.add_argument('--temp_value', '--temp_value', default=1.382, type=float,
                    metavar='temp_value', help='temp_value', dest='temp_value')
parser.add_argument('--epochs', '--epoch_number', default=200, type=int,
                    metavar='epochs', help='training epochs', dest='epochs')
parser.add_argument('--dp', '--data_path', default='..\\dataset\\amazon\\', type=str,
                    metavar='dp', help='the path of dataset', dest='dp')
parser.add_argument('--gpu', default=0, type=int,
                    metavar='GPU', help='gpu number to use', dest='gpu')
parser.add_argument('--bsz', default=10240, type=int,
                    metavar='bsz', help='batch_size', dest='bsz')
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
    filename = f'cr-no l3-{lr_params}-{con_weight_params}-{reg_weight_params}-{temp_value_params}-{bsz_params}.txt'

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

    hard_negatives = torch.load(base_path + 'co_items.pt')

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
                    test_gt,
                    hard_negatives)
    print('data is ready')
    cr_model = CR(args, data, filename)
    cr_model = cr_model.cuda()
    cr_model.train()


if __name__ == '__main__':
    main()

