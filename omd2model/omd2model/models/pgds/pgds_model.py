
import pyro.distributions as dist
import pyro
import torch
from pyro import poutine

from omd2model.modules import dir_helper


class PGDS():

    def __init__(self, K, A, band=2, emis_type="smd", trans_type="smd", prior={}):
        self.K = K
        self.A = A
        self.emis_type = emis_type
        self.trans_type = trans_type
        self.prior = prior
        self.emis_ka = dir_helper.select_matrix_dir(emis_type, K, A, band, site_name="emis_ka", event_dim=1,prior=prior)
        self.trans_kk = dir_helper.select_matrix_dir(trans_type, K, K, band, site_name="trans_kk", event_dim=1,prior=prior)
        print(f"PGDS – emis_type: {self.emis_type}, trans_type: {self.trans_type}\n{self.emis_ka.prior_name}:\n{self.emis_ka.prior}\n{self.trans_kk.prior_name}:\n{self.trans_kk.prior}")

    def get_prior(self, site_name, req_shape):
        if site_name in self.prior.keys() and list(self.prior[site_name].shape) == [*req_shape]:
            prior = self.prior[site_name]
            print(f"set prior {site_name} of shape {req_shape}")
        else:
            if site_name == "h/-1":
                prior = torch.ones(req_shape)
            elif site_name == "A":
                prior = torch.ones(req_shape)
            else:
                prior = torch.ones(req_shape)
        return prior

    def data_shapes(self, data, sites=["x"]):
        if "n_seq" in data.keys():
            n_seq = data["n_seq"]
            max_len = data["max_len"]
            data = {site: {"value": None, "mask": torch.ones(n_seq, self.A, max_len).bool()} for site in sites}
            print(f"generate data n_seq: {n_seq}, A: {self.A}, max_len: {max_len}")
        else:
            n_seq = data[sites[0]]["value"].shape[1]
            max_len = data[sites[0]]["value"].shape[-1]
        return data, n_seq, max_len

    ## ___________________________________________________________________
    ##
    ## MODEL
    ## ___________________________________________________________________
    def compute_rate_vva(self, h_vvk, emis_dir_ka):
        rate_vva = torch.einsum("...vwk,...ka->...vwa",(h_vvk, emis_dir_ka))
        return rate_vva

    def model(self, data, sites=["x"], event_dim=4):

        ## hyperparams
        epsilon = 1
        tau = 1
        data, self.V, self.T = self.data_shapes(data, sites)

        ## hyperpriors
        delta_a = pyro.sample("delta_a", dist.Gamma(self.get_prior("A", [self.A]), epsilon).to_event(0))

        ## priors
        trans_kk = self.trans_kk.sample()
        emis_ka = self.emis_ka.sample()

        h_vvk = pyro.sample("h/{}".format(-1),dist.Gamma(tau * self.get_prior("h/-1", [self.V, self.V, self.K]), tau).to_event(3))
        rate_svvat = torch.empty(h_vvk.shape[0], self.V, self.V, self.A, self.T)  ## S,V,V,A,T

        for t in pyro.markov(range(0, self.T)):
            h_vvk = torch.einsum("...vwk,...ck->...vwk",(h_vvk, trans_kk))
            h_vvk = pyro.sample("h/{}".format(t), dist.Gamma(tau * h_vvk, tau).to_event(3))
            rate_vva = self.compute_rate_vva(h_vvk, emis_ka)
            rate_svvat[:, :, :, :, t] = rate_vva.view(-1, self.V, self.V, self.A)

        ## likelihood
        with pyro.plate("V source", self.V, dim=-4):
            with pyro.plate("V target", self.V, dim=-3):
                with pyro.plate("A", self.A, dim=-2):
                    with pyro.plate("T", self.T, dim=-1):
                        #rate_svvat = rate_svvat.view(-1, self.V, self.V, self.A, self.T)
                        rate_svvat = torch.einsum("...a,...vwat->...vwat", (delta_a.view(-1, self.A), rate_svvat))
                        with pyro.poutine.mask(mask=data["x"]["mask"]):
                            x = pyro.sample("x", dist.Poisson(rate_svvat),
                            obs=data[sites[0]]["value"] if data[sites[0]]["value"] != None else None)


if __name__ == '__main__':

    hmm = PGDS(K=3, A=3, emis_type="smd", trans_type="smd")
    model = poutine.uncondition(hmm.model)
    trace = poutine.trace(model).get_trace({"n_seq": 100, "max_len": 4})
    print(trace.format_shapes())
