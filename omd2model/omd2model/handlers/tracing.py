import pyro, torch
from pyro import poutine
from pyro.infer import infer_discrete
import inspect
from omd0configs import gpu_config
from omd2model.handlers import utils
from tqdm import tqdm

def trace_handler(model, data, params_samples=None, infer=-1, event_dim=None, compute_log_prob=True, combine_seq=True, sample_plate=True):

    if not isinstance(event_dim, int):  ## get event dim
        event_dim = inspect.getfullargspec(model).defaults[-1]

    if isinstance(params_samples, dict):
        n_samples = list(params_samples.values())[0].shape[0]
        if n_samples > 1 and sample_plate == True:
            event_dim += 1
            sample_plate = pyro.plate("samples", n_samples, dim=-event_dim)
            model = sample_plate(model)
        model = poutine.condition(model, params_samples)

    elif isinstance(params_samples, int):
        n_samples = params_samples
        event_dim += 1
        sample_plate = pyro.plate("samples", n_samples, dim=-event_dim)
        model = sample_plate(model)
        model = poutine.uncondition(model)

    if infer >= 0: ## discrete latent variables
        event_dim += 1
        model = infer_discrete(model, first_available_dim=-event_dim, temperature=infer)

    trace = poutine.trace(model).get_trace(data)

    if compute_log_prob:
        site_filter = lambda name, site: True if "x" in name else False ## log_prob only of observed sites
        trace.compute_log_prob(site_filter = site_filter)

    if combine_seq:
        trace.nodes = utils.combine_trace_sites(trace.nodes)  ## only needed for time series model
        trace = add_trace_mask(trace, data)
    return trace


def partition_trace(model, data, params_samples=None, sample_batch=2, keep_sites=["x"]):
    n_samples = list(params_samples.values())[0].shape[0]
    batch_idx = torch.cat((torch.arange(0, n_samples, sample_batch), torch.tensor([n_samples]))).long()
    combined_trace_dict = {}

    for i in tqdm(range(0, len(batch_idx) - 1)):
        start_idx = batch_idx[i].item()
        end_idx = batch_idx[i + 1].item()
        params_sub = {k: v[start_idx:end_idx] for k, v in params_samples.items()}
        subtrace = trace_handler(model, data, params_samples=params_sub)
        for site, site_dict in subtrace.nodes.items():
            if site in keep_sites:
                combined_trace_dict[site + f"/{i}"] = {k: v.to("cpu") for k,v in site_dict.items() if isinstance(v, torch.Tensor)}
                gpu_config.empty_gpu_cache()
        #combined_trace_dict = {**combined_trace_dict,**{k + f"/{i}": v for k, v in subtrace.nodes.items() if k in keep_sites}}
    combined_dict = utils.combine_trace_sites(combined_trace_dict, stack_dim=[0])
    return combined_dict


def add_trace_mask(trace, data):
    for site in trace.nodes:
        if site in data.keys():
            if "mask" in data[site].keys():
                trace.nodes[site]["mask"] = data[site]["mask"]
    return trace



