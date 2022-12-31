import re
from collections import defaultdict
import torch
from pyro.poutine.indep_messenger import CondIndepStackFrame


def natural_keys(text):
    def atoi(text):  ## needed for human-like sorting "h/1", "h/2, "h/10"
        return int(text) if bool(re.search(r'-?\d', text)) else text
    return [atoi(c) for c in re.split(r'(-?\d+)', text)]


def combine_trace_site_vars(site_dict, stack_dim = -1):
    for k, v in site_dict.items():
        if len(site_dict[k]) > 0:
            if isinstance(site_dict[k][0], torch.Tensor):
                if isinstance(stack_dim, list) and len(site_dict[k][0].shape) > 0: ## concat at stack_dim
                    site_dict[k] = torch.cat(site_dict[k], dim=stack_dim[0])
                elif isinstance(stack_dim, int): ## stack over stack_dim
                    site_dict[k] = torch.stack(site_dict[k], dim=stack_dim)
                    if len(site_dict.keys()) == 1:  ## no need of sub-dict
                        return site_dict[k]
            elif k == "cond_indep_stack": ## increase event dim counter
                if len(site_dict[k][0]) > 0:
                    old_stack = site_dict[k][0][0] ## access CondIndepStackFrame.dim
                    site_dict[k] = (CondIndepStackFrame(old_stack.name, old_stack.dim - 1, old_stack.size, old_stack.counter),)
            elif k == "name":  ## change name
                site_dict[k] = site_dict[k][0].split("/")[0]
            else:  ## lists
                site_dict[k] = site_dict[k][0]
    return site_dict


def combine_trace_sites(site_data, stack_dim = -1):

    split_site = ""
    site_dict = defaultdict(list)  ## init new site_dict for new site
    sites = list(site_data.keys())
    sites.sort(key=natural_keys) ## sorting is requirement

    for i, site in enumerate(sites):
        #if "infer" in data[site].keys():
        #    if '_markov_scope' in data[site]["infer"]:
        site_list = site.split("/")
        if len(site_list) > 1:
            if site_list[0] != split_site: ## finish loop and pack
                if split_site != "":
                    site_dict = combine_trace_site_vars(site_dict, stack_dim)
                    site_data[split_site] = site_dict

                split_site = site_list[0] ## set new split site
                site_dict = defaultdict(list)  ## init new site_dict for new site

            if isinstance(site_data[site], dict) == False: ## value: tensor
                site_dict["value"].append(site_data[site]) ## copy over elements
            else:
                for k, v in site_data[site].items():  ## value: dict
                    site_dict[k].append(v) ## copy over elements
            del site_data[site]

    site_dict = combine_trace_site_vars(site_dict, stack_dim)
    site_data[split_site] = site_dict
    return site_data



if __name__ == '__main__':
    example = {"h/-1": torch.FloatTensor([0.2,0.3]), "h/1": torch.FloatTensor([0.2,0.3])}
    example = combine_trace_sites(example)
    print(example)

    example = {"h/-1": {"value": torch.FloatTensor([0.2,0.3]), "mask": torch.LongTensor([1,0])}, "h/1": {"value": torch.FloatTensor([0.2,0.3]), "mask": torch.LongTensor([1,0])}}
    example = combine_trace_sites(example)
    print(example)