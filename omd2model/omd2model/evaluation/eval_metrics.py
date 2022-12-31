import numpy as np
import torch
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error

from omd2model.evaluation import eval_helpers

def param_evalution(true_prior, posterior_mean, label_switch=True):
    for site in posterior_mean.keys():
        if site in true_prior.keys():
            a = posterior_mean[site].to("cpu")
            b = true_prior[site].to("cpu")
            wasserstein = wasserstein_distance(a.view(-1) / torch.sum(a.view(-1)), b.view(-1) / torch.sum(b.view(-1)))
            print(f"{site} wasserstein dist: {round(wasserstein, 3)}")
            if label_switch == False:
                js = np.mean(distance.jensenshannon(a / torch.sum(a), b / torch.sum(b)))
                print(f"{site} js: {round(js, 3)}")


def evalute_post_dens(log_prob_sn, mask=None, post_type=""):
    ### computes ppd or perplexity
    log_prob_n = torch.logsumexp(log_prob_sn, axis=0) - torch.log(torch.tensor(log_prob_sn.shape[0]))
    if isinstance(mask, torch.Tensor):
        log_prob_n = log_prob_n.unsqueeze(0)
        log_prob_n = torch.masked_select(log_prob_n, mask.bool())
    log_prob_n = log_prob_n.view(-1)
    ppd = torch.exp(torch.mean(log_prob_n, 0))  ## over data points
    print(f"{post_type} ppd: {round(float(ppd), 4)}")


def eval_point_pred(hat, gt, mask=None, gt_mask=None, post_type=""):
    ### computes mae
    hat = eval_helpers.get_mean_mode(hat)  ## mode / mean over samples
    if hat.shape[0] != gt.shape[0]:  ## is sample sides are different
        gt = eval_helpers.get_mean_mode(gt)  ## mean over latent sides
    if isinstance(mask, torch.Tensor) or isinstance(gt_mask, torch.Tensor):
        if isinstance(mask, torch.Tensor) and hat.shape == gt.shape:  ## impute case
            hat = hat[mask.bool()]
            gt = gt[mask.bool()]
        else: ## split case
            if isinstance(mask, torch.Tensor):
                if mask.shape[-1] < hat.shape[-1]: ## latent h case
                    mask = torch.cat((torch.zeros(1, mask.shape[-2], 1), mask), axis = -1)
                hat = hat[mask.bool()]
            if isinstance(gt_mask, torch.Tensor):
                if gt_mask.shape[-1] < gt.shape[-1]:  ## latent h case
                    gt_mask = torch.cat((torch.zeros(1, gt_mask.shape[-2], 1), gt_mask), axis = -1)
                gt = gt[gt_mask.bool()]
    hat = hat.view(-1).to("cpu")
    gt = gt.view(-1).to("cpu")
    rmse = mean_absolute_error(gt, hat)
    print(f"{post_type} mae: {round(float(rmse), 4)}")







