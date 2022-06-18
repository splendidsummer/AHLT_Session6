import torch

if __name__ == '__main__':
    a = torch.tensor([[1, 2], [1, 2], [2, 3]])
    a.size(-1)
    torch.sqrt(torch.tensor(2))
    print(111)

    from torch.nn.utils.rnn import pad_sequence
    import torch

    a = torch.randn((3, 4, 3))
    b = torch.randn((3, 4, 3))
    c = torch.randn((3, 4, 3))

    a = torch.cat([a, b, c], dim=-1)
    lens = torch.tensor([3, 2, 1])
    a = a[range(len(a)), lens-1, :]
    print(a.shape)


    print(222)

