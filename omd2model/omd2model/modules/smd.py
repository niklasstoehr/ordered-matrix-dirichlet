
import pyro
import torch
import pyro.distributions as dist


class SMD:

    def __init__(self, K, A, priors={}, site_name= "phi_ka", event_dim=1, device="cpu"):

        self.K = K
        self.A = A
        self.device = device
        self.site_name = site_name
        self.priors = priors
        self.event_dim = event_dim
        self.prior = self.select_prior(priors)
        self.dir_type = "SMD"

    def sample(self, prior = None, n_samples = 1):
        prior = self.shape_prior(prior)
        if n_samples > 1:
            prior = torch.repeat_interleave(prior.unsqueeze(0), n_samples, dim=0)
        phi_ka = pyro.sample(self.site_name, dist.Dirichlet(prior).to_event(self.event_dim))
        return phi_ka


    def select_prior(self, prior):
        self.prior_name = self.site_name + "_SMD"
        if self.prior_name in prior.keys():
            prior = self.shape_prior(prior[self.prior_name].to(self.device))
        else:  ## default prior
            prior = torch.ones(self.K, self.A, device = self.device) ## set uniform prior
        return prior


    def shape_prior(self, prior):
        if prior == None:
            prior = self.prior
        elif list(prior.shape[-2:]) == [self.K, self.A]:
            prior = prior.float()  ## normal case
        elif prior.shape[-1] == self.A:
            prior = torch.repeat_interleave(prior.float().unsqueeze(-2), self.K, dim=-2)
        elif prior.shape[-1] == 1:
            prior = torch.repeat_interleave(prior.float(), self.K, dim=-1)
            prior = torch.repeat_interleave(prior.unsqueeze(-1), self.A, dim=-1)
        else:
            assert False, f"{self.prior_name} should be of shape {self.K, self.A}"
        return prior



##_________________________________________________________________________


if __name__ == '__main__':

    K = 3
    A = 3

    normalDir_site = SMD(K, A, site_name="phi_ka", priors={"phi_ka_smd": torch.tensor(([800, 10, 10]))}, event_dim=1)
    print("beta_a", normalDir_site.prior)
    print("sample", normalDir_site.sample().shape)

    normalDir_site = SMD(K, A, site_name="phi_ka", priors={"phi_ka_smd": torch.tensor([10])}, event_dim=1)
    print("phi_ka_smd", normalDir_site.prior)
