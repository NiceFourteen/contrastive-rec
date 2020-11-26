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


def test_positive_loss():
    item_set = torch.load('..\\dataset\\amazon\\item_set.pt')
    user_set = torch.load('..\\dataset\\amazon\\user_set.pt')
    item_embedding = torch.nn.Embedding(len(user_set), 64)
    item_embedding = torch.nn.init.normal_(item_embedding.weight, std=0.01)
    item_tensor = torch.from_numpy(item_set)
    iev = (item_embedding[1:10, :]).double()
    ifv = (item_tensor[1:10, :]).double()
    loss = torch.einsum('ij, ij -> i', [ifv, iev])
    loss_exp = torch.exp(loss)
    print(loss_exp)
    return loss_exp


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def test_cat():
    tensor_a = torch.randn(5, 3)
    tensor_b = torch.randn(9, 3)
    ten = tensor_a - tensor_b
    ten_cat = torch.cat([tensor_a, tensor_b], dim=0)
    a_norm = torch.pow(tensor_a, 2)
    print(a_norm)


if __name__ == '__main__':
    """"""
    x = torch.randn(2, 3)
    x_pos = torch.randn(2, 3)
    x_u = x.unsqueeze(1)
    y = torch.randn(4, 3)

    z = x_u - y
    # result1 = torch.einsum('ni, ni -> n', [x, x_pos])
    results = torch.einsum('nij, nij -> ni', [z, z])
    result_sum = torch.sum(results, dim=1)
    print(results)
    print(result_sum)


