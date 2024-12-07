import pyro.distributions as dist
from pyro.ops.indexing import Vindex
import pyro
import torch
import numpy as np
from pyro import poutine
from omd0configs import gpu_config
from omd2model.modules import dir_helper


class HMM():

    def __init__(self, K, A, emis_type="smd", trans_type="smd", prior={}):
        self.K = K
        self.A = A
        self.emis_ka = dir_helper.select_matrix_dir(emis_type, K, A, site_name="emis_ka", prior=prior)
        self.trans_kk = dir_helper.select_matrix_dir(trans_type, K, K, site_name="trans_kk", prior=prior)
        self.alpha_k = self.set_state_prior(prior)
        if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print("using device:", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        print(f"HMM – emis_type: {emis_type}, trans_type: {trans_type}\n{self.emis_ka.prior_name}:\n{self.emis_ka.prior}\n{self.trans_kk.prior_name}:\n{self.trans_kk.prior}\nalpha_k:\n{self.alpha_k}")

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

            synth_data = {}
            for site in sites:
                if site in data.keys():
                    mask = data[site]["mask"] ## copy over mask
                else:
                    mask = torch.ones((n_seq, max_len)).bool() ## synth mask
                synth_data[site] = {"value": None, "mask": mask}
            data = synth_data
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
        emis_dir_ka = self.emis_ka.sample()
        h_alpha_k = pyro.sample("h_alpha_k", dist.Dirichlet(self.alpha_k).to_event(0))

        with pyro.plate("seq", self.V, dim=-1):
            h = pyro.sample("h/{}".format(-1), dist.Categorical(h_alpha_k).to_event(0))
            for t in pyro.markov(range(self.T)):
                h = pyro.sample("h/{}".format(t), dist.Categorical(Vindex(trans_dir_kk)[..., h.long(), :]).to_event(0),infer={"enumerate": "parallel"})
                with pyro.poutine.mask(mask=data[sites[0]]["mask"][..., t]):
                    x = pyro.sample("{}/{}".format(sites[0], t),dist.Categorical(Vindex(emis_dir_ka)[..., h.long(), :]),
                    obs=data[sites[0]]["value"][..., t] if data[sites[0]]["value"] != None else None)

if __name__ == '__main__':

    hmm = HMM(K=3, A=3, emis_type="smd", trans_type="smd")
    model = poutine.uncondition(hmm.model)
    trace = poutine.trace(model).get_trace({"n_seq": 100, "max_len": 4})
    print(trace.format_shapes())
