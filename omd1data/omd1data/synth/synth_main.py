
from omd0configs.viz import vizuals
from omd1data.synth.synth_functions import function
from omd1data.synth.synth_post import make_poisson, normalize_time_series, add_random_noise

import torch
import matplotlib.pyplot as plt


def generate_sequence(V = 100, A = 20, T = 20, seq_fn="linear", seq_kwargs={"noise": 0}, cnt_fn="gamma", cnt_kwargs={"poisson": 5, "noise": 0}, vis=-1):
    Y = torch.zeros(V, A, T).long()

    for v in range(0, V):
        ## get sequence and counts_________________
        seq_y = function(A, T, fn=seq_fn, kwargs=seq_kwargs)

        cnt_y = function(2, T, fn=cnt_fn, kwargs=cnt_kwargs)
        cnt_y = make_poisson(y=cnt_y, kwargs=cnt_kwargs)
        cnt_y = normalize_time_series(y=cnt_y, kwargs=cnt_kwargs)

        ## add random noise_________________
        seq_y = add_random_noise(seq_y, seq_kwargs)
        cnt_y = add_random_noise(cnt_y, cnt_kwargs)

        ## build count tensor________________
        y = torch.zeros(A, T).long()
        seq_y = torch.floor(seq_y).long()
        cnt_y = torch.round(cnt_y).long()

        indeces = torch.arange(0, T)
        y[seq_y[indeces], indeces] = cnt_y[indeces]
        Y[v, :, :] = y

    if  vis >= 0:
        vizuals.visualize_2D(Y, i=vis)
        plt.show()
    return Y




if __name__ == '__main__':


    V = 1  ## source-target pairs
    A = 5  ## action types
    T = 10  ## time steps

    seq_fn = "linear"
    seq_kwargs = {"noise": 0, "dir": "up", "noise": 2}

    cnt_fn = "constant"
    cnt_kwargs = {"total_count": 1000, "noise": 1000}


    Y = generate_sequence(V, A, T, seq_fn=seq_fn, seq_kwargs=seq_kwargs, cnt_fn=cnt_fn, cnt_kwargs=cnt_kwargs, vis = 0)
    print(Y)