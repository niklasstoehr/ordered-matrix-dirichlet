
import torch
import pyro
import numpy as np
import pyro.distributions as dist
from pyro.ops.indexing import Vindex
from pyro import poutine

from omd2model.modules import dir_helper


class LDA():

    def __init__(self, n_c, n_voc, phi_dir_type="smd", prior={}):
        self.n_c = n_c
        self.n_voc = n_voc
        self.phi_dir_type = phi_dir_type
        self.alpha_c = self.set_topic_prior(prior)
        self.phi_dir = dir_helper.select_matrix_dir(phi_dir_type, n_c, n_voc, site_name="trans_kk", prior=prior)
        print(f"LDA – phi_dir: {self.phi_dir_type}, n_c: {self.n_c}, n_voc: {self.n_voc}\nalpha_c: {self.alpha_c}\n{self.phi_dir.prior_name}:\n{self.phi_dir.prior}\n")

    def set_topic_prior(self, prior, normalize=True):
        if "alpha_c" in prior.keys():
            if prior["alpha_c"].squeeze().shape[0] == self.n_c:
                ## topic prior provided
                topic_prior = prior["alpha_c"]
            else:  ## class labels provided
                y = np.array(prior["alpha_c"])
                _, y_prior = np.unique(y, return_counts=True)
                if normalize:
                    y_prior = y_prior / np.sum(y_prior)
                topic_prior = torch.tensor(y_prior).view(-1)
        else:  ## set up uniform prior
            topic_prior = torch.ones(self.n_c)
        return topic_prior.float()

    def data_shapes(self, data, sites=["x"]):
        ## get data shape

        if "n_docs" in data.keys():
            n_docs = data["n_docs"]
            n_tokens = data["n_tokens"]
            data = {str(site): {"value": None, "mask": torch.ones((n_docs, n_tokens)).bool()} for site in sites}
            print(f"generate data n_docs: {n_docs}, n_tokens: {n_tokens}")
        else:
            n_docs = data[sites[0]]["value"].shape[1]
            n_tokens = data[sites[0]]["value"].shape[-1]
        return data, n_docs, n_tokens

    ## ___________________________________________________________________
    ##
    ## MODEL
    ## ___________________________________________________________________

    def model(self, data, sites=["x"], event_dim=2):

        data, n_docs, n_tokens = self.data_shapes(data, sites)
        topic_word_ck = self.phi_dir.sample()
        with pyro.plate("n_docs", n_docs, dim=-2):
            pi_z_c = pyro.sample("pi_c", dist.Dirichlet(self.alpha_c))
            with pyro.plate("n_tokens", n_tokens, dim=-1):
                z_c = pyro.sample('z', dist.Categorical(pi_z_c).to_event(0), infer={"enumerate": "parallel"})
                topic_k = (Vindex(topic_word_ck)[..., z_c.long(), :])
                with pyro.poutine.mask(mask=data[sites[0]]["mask"]):
                    tokens = pyro.sample(sites[0], dist.Categorical(topic_k),
                    obs=data[sites[0]]["value"])
        return tokens


if __name__ == '__main__':

    lda = LDA(n_c = 3, n_voc = 3, phi_dir_type = "smd")
    model = poutine.uncondition(lda.model)
    trace = poutine.trace(model).get_trace({"n_docs": 100, 'n_tokens': 10})
    print(trace.format_shapes())

