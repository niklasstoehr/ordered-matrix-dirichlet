
from pyro.infer.autoguide import initialization as mcmc_inits
from pyro.infer import MCMC, NUTS, HMC, Predictive
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from omd2model.handlers import generation, tracing, utils


def mcmc_init_params(model, data, num_samples=10):
    p = Predictive(model, num_samples=num_samples)
    init_params = p(data)
    init_params = {k: torch.mean(v.float(), dim=0) for k, v in init_params.items()}
    return init_params


def run_mcmc(model, data, num_samples=100, warmup_steps=50, max_tree_depth=5, kernel="NUTS", print_summary=False):
    init_params = mcmc_init_params(model, data)
    if kernel == "NUTS":
        mcmc_kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=max_tree_depth,init_strategy=mcmc_inits.init_to_value(values=init_params),jit_compile=False)
    elif kernel == "HMC":
        mcmc_kernel = HMC(model, step_size=1, num_steps=max_tree_depth,init_strategy=mcmc_inits.init_to_value(values=init_params),jit_compile=False)
    mcmc = MCMC(mcmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
    mcmc.run(data)
    if print_summary:
        print(mcmc.summary(prob=0.8))
    return mcmc


def get_post_mean(post_params, model, data, gt_params = None, combine_seq = True, plot_params=False):
    posterior_mean = dict()

    if gt_params != None:
        diff_dict = [k for k in gt_params if k not in post_params.keys()]
        if len(diff_dict) > 0:
            post_params = utils.combine_trace_sites(post_params)
            predictive = Predictive(model, post_params)
            sites = predictive(data)
            post_params = {k: v for k, v in post_params.items() if k in gt_params}
            sites = {k:v for k,v in sites.items() if k in diff_dict}
            post_params = {**post_params, **sites}

    for site, values in post_params.items():
        if values.is_floating_point(): ## only continuous sites
            values = values.mean(0)
            if values.shape[0] == 1:
                values = values.squeeze()
            else:
                values = values.mean(0).squeeze()
            posterior_mean[site] = values

    if combine_seq:
        posterior_mean = utils.combine_trace_sites(posterior_mean)

    if plot_params:
        for k, v in posterior_mean.items():
            print(f"{str(k)}\n{v}")
    return posterior_mean


def infer_post_discrete(model, data, post_samples, infer=0, combine_seq=False, obs_site="x"):
    trace = tracing.trace_handler(model, data, params_samples=post_samples, infer=infer, combine_seq=combine_seq)
    _, all_params = generation.trace_to_sites_params(trace, sites=[], mean_params=False)
    all_params = {k: v for k,v in all_params.items() if not k.startswith(obs_site)} ## delete observed sites
    return all_params