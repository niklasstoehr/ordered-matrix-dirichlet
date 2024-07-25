import pyro.distributions as dist
import pyro
import torch
from pyro import poutine
from omd0configs import gpu_config
from omd2model.modules import dir_helper


class PGDS_BPT():

    def __init__(self, K, A, C, band=3, emis_type="smd", trans_type="smd", prior={}, select_gpu=0):
        self.K = K
        self.A = A
        self.C = C
        self.band = band
        self.emis_type = emis_type
        self.trans_type = trans_type
        self.prior = prior
        self.set_up(select_gpu)

    def set_up(self, select_gpu):
        self.device = gpu_config.get_device(select_gpu=select_gpu)
        self.emis_ka = dir_helper.select_matrix_dir(self.emis_type, self.K, self.A, self.band, site_name="emis_ka",event_dim=1, prior=self.prior, device=self.device)
        self.trans_kk = dir_helper.select_matrix_dir(self.trans_type, self.K, self.K, self.band, site_name="trans_kk", event_dim=1, prior=self.prior, device=self.device)
        print(f"PGDS BPT {self.device} – emis_type: {self.emis_type}, trans_type: {self.trans_type}\n{self.emis_ka.prior_name}:\n{self.emis_ka.prior}\n{self.trans_kk.prior_name}:\n{self.trans_kk.prior}")

    def get_prior(self, site_name, req_shape):
        if site_name in self.prior.keys() and list(self.prior[site_name].shape) == [*req_shape]:
            prior = self.prior[site_name].to(self.device)
        else:
            if site_name == "h/-1":
                prior = torch.ones(req_shape, device=self.device)
            elif site_name == "t/-1":
                prior = torch.ones(req_shape, device=self.device)
            elif site_name == "V":
                prior = torch.ones(req_shape, device=self.device)
            elif site_name == "A":
                prior = torch.ones(req_shape, device=self.device)
            else:
                prior = torch.ones(req_shape, device=self.device)
        return prior

    def data_shapes(self, data, sites=["x"]):
        if 'n_seq' in data.keys():
            n_seq = data["n_seq"]
            max_len = data["max_len"]
            synth_data = {}
            for site in sites:
                if site in data.keys():
                    mask = data[site]["mask"]  ## copy over mask
                else:
                    mask = torch.ones(1, n_seq, n_seq, self.A, max_len).bool()  ## synth mask
                synth_data[site] = {"value": None, "mask": mask}
            data = synth_data
            print(f"generate data n_seq V: {n_seq}, n_seq V: {n_seq}, A: {self.A}, max_len T: {max_len}")
        else:
            n_seq = data[sites[0]]["value"].shape[1]
            max_len = data[sites[0]]["value"].shape[-1]
        return data, n_seq, max_len

    ## ___________________________________________________________________
    ##
    ## MODEL
    ## ___________________________________________________________________

    def compute_rate_vva(self, h_cck, affil_vc, emis_dir_ka):
        #h_vck = torch.einsum("...vc,...cbk->...vbk", (affil_vc, h_cck))
        h_cvk = affil_vc @ h_cck  ## "...vc,...cbk->...vbk"
        #h_vvk = torch.einsum("...wc,...vck->...vwk", (affil_vc, h_vck))
        h_kvv = affil_vc @ torch.movedim(h_cvk, -1, -3)  ## "...wc,...vck->...vwk"
        #rate_vva = torch.einsum("...ka,...vwk->...vwa", (emis_dir_ka, h_vvk))
        rate_vva = torch.movedim(h_kvv, -3, -1) @ emis_dir_ka  ## "...ka,...vwk->...vwa"
        return rate_vva

    def model(self, data, sites=["x"], event_dim=4):

        data, self.V, self.T = self.data_shapes(data, sites)
        tau = 1
        ## sample | event | latent --> 1 | 4 | 3
        VC_prior_conc = torch.repeat_interleave(self.get_prior("V", [self.V]).unsqueeze(-1), self.C, dim=-1)
        VC_prior_rate = torch.ones(self.V, self.C, device=self.device)
        affil_vc = pyro.sample("affil_vc", dist.Gamma(VC_prior_conc, VC_prior_rate).to_event(2))

        delta_a = pyro.sample("delta_a", dist.Gamma(self.get_prior("A", [self.A]).unsqueeze(-1), torch.ones(self.A, device=self.device).unsqueeze(-1)).to_event(0))

        trans_dir_kk = self.trans_kk.sample()
        emis_dir_ka = self.emis_ka.sample()

        h_tcck = [pyro.sample("h/{}".format(-1),dist.Gamma(tau * self.get_prior("h/{}".format(-1), [self.C, self.C, self.K]),tau).to_event(2))]
        delta_t = [pyro.sample("delta_t/{}".format(-1), dist.Gamma(tau, tau).to_event(0))]

        for t in pyro.markov(range(0, self.T)):
            h_tcck.append(pyro.sample("h/{}".format(t), dist.Gamma(tau * (h_tcck[t] @ trans_dir_kk), tau).to_event(2)))
            delta_t.append(pyro.sample("delta_t/{}".format(t), dist.Gamma(tau * delta_t[t], tau).to_event(0)))

        ## likelihood V-V-A-T
        h_tcck = torch.stack(h_tcck[1:], dim=0)
        delta_t = torch.stack(delta_t[1:], dim=-1)
        with pyro.plate("V source", self.V, dim=-4):
            with pyro.plate("V target", self.V, dim=-3):
                with pyro.plate("A", self.A, dim=-2):
                    with pyro.plate("T", self.T, dim=-1):
                        with pyro.poutine.mask(mask=data["x"]["mask"].to(self.device)):
                            ## tensor decomposition
                            rate_tvva = self.compute_rate_vva(h_tcck, affil_vc, emis_dir_ka)
                            rate_vvat = torch.movedim(rate_tvva, 0, -1).view(-1, self.V, self.V, self.A, self.T)
                            rate_vvat = delta_a.view(-1, 1, 1, self.A, 1) * rate_vvat
                            rate_vvat = delta_t.view(-1, 1, 1, 1, self.T) * rate_vvat
                            x = pyro.sample("x", dist.Poisson(rate_vvat).to_event(0),obs=data["x"]["value"].to(self.device) if data["x"]["value"] != None else None)

if __name__ == '__main__':

    pgds_bpt = PGDS_BPT(C=3, K=3, A=3, emis_type="smd", trans_type="smd")
    model = poutine.uncondition(pgds_bpt.model)
    trace = poutine.trace(model).get_trace({"n_seq": 100, "max_len": 4})
    print(trace.format_shapes())
