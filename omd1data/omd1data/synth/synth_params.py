import torch.nn.functional as F
import torch


def generate_params(V=3, C=3, K=3, T=10, epsilon=0.0001):
    params = {}

    C_index = torch.repeat_interleave(torch.arange(0, C), int(V / C))
    C_index_pad = F.pad(C_index, (0, V - C_index.shape[0]), "constant", C_index[-1])

    params["source_vc"] = (F.one_hot(C_index_pad) + epsilon).unsqueeze(0)
    params["target_vc"] = (F.one_hot(C_index_pad) + epsilon).unsqueeze(0)
    params["h--1"] = (F.one_hot(torch.randint(low=0, high=K, size=(C ** 2,)).view(C, C)).float() + epsilon) * K
    params["delta_t"] = torch.ones(T) * 1
    return params


if __name__ == '__main__':
    params = generate_params(V=3, C=3, K=3, T=10)
    print(params)


