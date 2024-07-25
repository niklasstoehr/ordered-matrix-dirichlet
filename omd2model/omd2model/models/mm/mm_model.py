
import pyro.distributions as dist
from pyro.ops.indexing import Vindex
import pyro
import torch
import numpy as np
from pyro import poutine
from omd2model.modules import dir_helper

import pyro.distributions as dist
from pyro.ops.indexing import Vindex
import pyro
import torch
import numpy as np
from pyro import poutine

from omd2model.modules import dir_helper


class MM():

    def __init__(self, K, trans_type="smd", prior={}):
        self.K = K
        self.trans_kk = dir_helper.select_matrix_dir(trans_type, K, K, site_name="trans_kk", prior=prior)
        self.alpha_k = self.set_state_prior(prior)
        print(
            f"MM – trans_type: {trans_type}\n{self.trans_kk.prior_name}:\n{self.trans_kk.prior}\nalpha_k:\n{self.alpha_k}")

    def set_state_prior(self, prior, normalize=True):
        if "alpha_k" in prior.keys():
            if prior["alpha_k"].squeeze().shape[0] == self.K:
                ## state prior provided
                state_prior = prior["alpha_k"]
            else:  ## state labels provided
                y = np.array(prior["alpha_k"])
                _, state_prior = np.unique(y, return_counts=True)
                if normalize:
                    state_prior = state_prior / np.sum(state_prior)
                state_prior = torch.tensor(state_prior).view(-1)
        else:  ## set up uniform prior
            state_prior = torch.ones(self.K)  # * 100
        return state_prior.float()

    def data_shapes(self, data, sites=["x"]):
        if "n_seq" in data.keys():
            n_seq = data["n_seq"]
            max_len = data["max_len"]
            data = {site: {"value": None, "mask": torch.ones((n_seq, max_len)).bool()} for site in sites}
            print(f"generate data n_seq: {n_seq}, max_len: {max_len}")
        else:
            n_seq = data[sites[0]]["value"].shape[1]
            max_len = data[sites[0]]["value"].shape[-1]
        return data, n_seq, max_len

    ## ___________________________________________________________________
    ##
    ## MODEL
    ## ___________________________________________________________________

    def model(self, data, sites=["x"], event_dim=1):

        data, self.V, self.T = self.data_shapes(data, sites)
        trans_dir_kk = self.trans_kk.sample()
        h_alpha_k = pyro.sample("h_alpha_k", dist.Dirichlet(self.alpha_k).to_event(0))

        with pyro.plate("seq", self.V, dim=-1):
            x = pyro.sample("h/{}".format(-1), dist.Categorical(h_alpha_k).to_event(0), infer={"enumerate": "parallel"})
            for t in pyro.markov(range(self.T)):
                with pyro.poutine.mask(mask=data[sites[0]]["mask"][..., t]):
                    x = pyro.sample("{}/{}".format(sites[0], t),dist.Categorical(Vindex(trans_dir_kk)[..., x.long(), :]).to_event(0), infer={"enumerate": "parallel"},
                    obs=data[sites[0]]["value"][..., t] if data[sites[0]]["value"] != None else None)


if __name__ == '__main__':
    hmm = MM(K=3, trans_type="smd")
    model = poutine.uncondition(hmm.model)
    trace = poutine.trace(model).get_trace({"n_seq": 20, "max_len": 4})
    print(trace.format_shapes())

