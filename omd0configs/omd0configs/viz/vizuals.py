import numpy as np
import torch
import pyro.distributions as dist
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib as mpl

from omd0configs.viz.viz_helpers import reshape_tensor

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def visualize_dir(X_KA, xlabel='action type $a$', ylabel="$\phi_{a", title="", fontsize = 15, fig_size=(7, 4), subfig = None):
    linewidth = 2
    markersize = 10

    X_KA = reshape_tensor(X_KA.to("cpu"), req_shape_len=2)
    K = X_KA.shape[0]
    color = cm.coolwarm(np.linspace(0.0, 1.1, K))

    if subfig == None:
        fig, axes = plt.subplots(K, 1, sharex=False, sharey=False, figsize=fig_size)
        plt.subplots_adjust(left=0.5, bottom=0.1, right=0.9, top=0.9, wspace=0.15, hspace=0.4)
        plt.suptitle(title, fontsize=fontsize, x=0.5, y=.95, ha='center')
    else:
        axes = subfig.subplots(K, 1, sharex=False, sharey=False)

    for k in range(K):
        row = k
        ax = axes[row]
        markerline, stemline, baseline, = ax.stem(np.arange(X_KA[k].shape[-1]), X_KA[k], markerfmt="o-b", linefmt="b-",
                                                  basefmt="w", use_line_collection=True)
        plt.setp(stemline, linewidth=linewidth, color ="black")
        plt.setp(markerline, markersize=markersize, color = color[k])
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        label = ylabel + f'{k + 1}' + '}$'
        ax.set_xticks([])
        if subfig == None:
            ax.set_ylabel(label, rotation=0, labelpad=20, fontsize=fontsize + 2)
        else:
            ax.set_yticks([round(torch.mean(X_KA[k]).item(),1)])
            if k == K-1:
                subfig.text(-0.05, 0.5, ylabel, va='center', fontsize = fontsize, rotation='vertical')

        top = torch.max(X_KA[k]) + 0.25
        bottom = torch.min(X_KA[k]) - 0.05
        if torch.std(X_KA[k]) <= 0.1:
            top = torch.max(X_KA[k]) + 0.5
        ax.set_ylim(bottom, top)

        if k == K - 1:
            ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xticks(np.arange(X_KA[-1].shape[-1]))




def visualize_2D(Y, i = 0, xlabel = "x", ylabel = "y", title = "", fontsize = 15, fig_size=(12, 5), show_cbar = True, square=False, subfig = None):

    Y = reshape_tensor(Y.to("cpu"), req_shape_len=3)

    if subfig == None:
        fig, ax = plt.subplots(figsize=fig_size)
        plt.suptitle(title, fontsize=fontsize, x=0.1, y=.95, ha='left')
    else:
        ax = subfig.subplots(1, 1, sharex=False, sharey=False)

    sns.heatmap(Y[i].numpy(), ax=ax, cmap="Blues", square=square, cbar=show_cbar)
    if show_cbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    color = cm.coolwarm(np.linspace(0.0, 1.1, Y[0].shape[0]))
    for i, tick_label in enumerate(ax.get_yticklabels()):
        tick_label.set_color(color[i])



def visualize_cdf(X_KA, xlabel="action type $a$", ylabel="$CDF_k(a)$", title="", fontsize = 15, fig_size=(7, 4), subfig = None):

    markersize = 15
    linewidth = 3

    X_KA = reshape_tensor(X_KA.to("cpu"), req_shape_len=2).numpy()
    K, A = X_KA.shape[0], X_KA.shape[1]
    C_KA = np.array([np.cumsum(X_KA[k]) for k in range(K)])
    color = cm.coolwarm(np.linspace(0.0, 1.1, K))

    if subfig == None:
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=fig_size)
        plt.suptitle(title, x=0.1, y=.95, fontsize=fontsize, ha='left')
    else:
        ax = subfig.subplots(1, 1, sharex=False, sharey=False)

    for k in range(K):
        ax.plot(np.arange(A), C_KA[k], linestyle=':', color='lightgrey', lw=linewidth,
                marker=f'${k + 1}$', markersize=markersize, markerfacecolor=color[k], markeredgewidth=0.4,
                markeredgecolor=color[k])
        ax.set_xticks(np.arange(A), fontsize=fontsize)
        ax.set_xticklabels(labels=np.arange(A), fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(0, 1.1)


def visualize_tensors(params, viz_sites=["phi_ck", "pi_c"], viz_type=""):
    for k, v in params.items():
        if k in viz_sites:
            if len(v.shape) == 1:
                visualize_2D(v, title = str(k) + " – "+ str(viz_type), fig_size=(6, 4))
            if len(v.shape) >= 2:
                visualize_dir(v, title = str(k) + " – "+ str(viz_type), fig_size=(9, 4))
                visualize_2D(v, title = str(k) + " – "+ str(viz_type), fig_size=(6, 4))


def trajectory_plot(data, jitter=0.15, title="", xlabel = 'timestep', ylabel = 'state', fig_size=(6, 3)):

    data = reshape_tensor(data.to("cpu"), req_shape_len=2)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    left_side = ax.spines["left"]
    left_side.set_visible(False)
    top_side = ax.spines["bottom"]
    top_side.set_visible(False)

    fontsize = 12
    plt.suptitle(title, x=0.1, y=.95, fontsize=fontsize, ha='left')
    color = cm.coolwarm(np.linspace(0.3, 0.9, torch.max(data) + 1))

    for i, seq in enumerate(data):
        y = data[i, :].numpy()
        scale = np.random.normal(loc=0.0, scale=jitter, size=len(y))
        ax.plot(np.arange(0, len(y)), y + scale, color=color[y[0]])

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(xlabel, fontsize=fontsize)

    plt.show()



if __name__ == '__main__':

    A = 20
    K = 10
    X_KA = dist.Dirichlet(torch.ones(K, A)).sample()
    visualize_dir(X_KA)

    T = 10
    Y = torch.rand(1, 2, T)
    visualize_2D(Y, i=0)