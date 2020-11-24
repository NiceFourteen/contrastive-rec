import torch


def test_sequential():
    item_fea = torch.load('..\\dataset\\amazon\\item_set.pt')
    item_tensor = torch.from_numpy(item_fea)
    item_tensor = item_tensor.float()
    item_tensor = item_tensor.cuda()
    dim_mlp = item_tensor.shape[1]
    item_mlp = torch.nn.Sequential(torch.nn.Linear(dim_mlp, dim_mlp), torch.nn.LeakyReLU())
    item_mlp = item_mlp.cuda()
    i = item_mlp(item_tensor)


def test_ein():
    matrix1 = torch.randn(1, 2)
    matrix2 = torch.randn(4, 2)
    matrix3 = torch.randn(1, 2)
    result_temp1 = torch.einsum('nc, kc -> nk', [matrix1, matrix2])
    result_temp2 = torch.einsum('nc, nc -> n', [matrix1, matrix3])
    result = torch.cat([result_temp2, result_temp1], dim=0)
    print(result)


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    """"""
    matrix = torch.randn(100, 9)
    batch_size_this = matrix.shape[0]
    ids = torch.randperm(batch_size_this)
    x_gather = concat_all_gather(matrix)
    batch_size_all = x_gather.shape[0]
