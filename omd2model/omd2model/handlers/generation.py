import torch
import pyro
from collections import defaultdict

from omd2model.handlers import tracing
from omd2model.evaluation import  eval_helpers


def trace_to_sites_params(trace, sites=["x"], mean_params = True):
    data_dict = defaultdict(dict)
    params_dict = dict()

    for k in trace.nodes.keys():
        if "value" in trace.nodes[k].keys() and "is_observed" in trace.nodes[k].keys():
            if k in sites: ## sites
                data_dict[k]["value"] = trace.nodes[k]["value"]#.squeeze()
                data_dict[k]["mask"] = torch.ones(trace.nodes[k]["value"].shape).bool()#.squeeze()
                data_dict[k]["log_prob"] = trace.nodes[k]["log_prob"]#.squeeze()

            else:  ## extract parameters
                if isinstance(trace.nodes[k]['fn'], pyro.poutine.subsample_messenger._Subsample) != True:
                    #if len(trace.nodes[k]['infer']) == 0:  ## filter out Z
                    param_value = trace.nodes[k]["value"]
                    if mean_params:
                        if param_value.shape != param_value.squeeze().shape:
                            param_value = eval_helpers.get_mean_mode(param_value)
                    params_dict[k] = param_value
    return data_dict, params_dict


def generate_data(model, data_stats, infer=-1, event_dim=None, sites=[], combine_seq=True):
    trace = tracing.trace_handler(model, data_stats, params_samples=1, infer=infer, event_dim=event_dim, combine_seq=combine_seq)
    data, params = trace_to_sites_params(trace, sites= sites)
    return data, params, trace


if __name__ == '__main__':
    pass
    #gt_hmm = hmm_model.HMM(K=3, A=3, emis_type="normdir", trans_type="normdir", prior={})
    #syth_data = {"n_seq": 100, "max_len": 4}
    #data, params, trace = generate_data(gt_hmm.model, syth_data, infer=0, sites=["x", "h"])

