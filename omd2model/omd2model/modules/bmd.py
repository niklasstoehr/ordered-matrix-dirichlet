import torch
import pyro
import pyro.distributions as dist


class BMD:

    def __init__(self, K, band=3, priors={}, site_name="pi_kk", event_dim=1, device="cpu"):

        assert K >= band, "need K >= band"
        if band <= 2:
            band = 3
            print(f"band needs to be >= 3")

        self.K = K
        self.band = band
        self.device = device
        self.site_name = site_name
        self.priors = priors
        self.event_dim = event_dim
        self.prior = self.select_prior(priors)
        self.dir_type = "bmd"


    def sample(self, prior = None, n_samples = 1):
        prior = self.shape_prior(prior)
        if n_samples > 1:
            prior = torch.repeat_interleave(prior.unsqueeze(0), n_samples, dim=0)
        pi_kband = banded_diagonal(band = prior, device = self.device)
        pi_kk = pyro.sample(self.site_name, dist.Dirichlet(pi_kband).to_event(self.event_dim))
        return pi_kk


    def select_prior(self, prior):
        self.prior_name = self.site_name + "_bmd"
        if self.prior_name in prior.keys():
            prior = self.shape_prior(prior[self.prior_name].to(self.device))
        else:  ## default prior
            prior = torch.ones(self.K, self.band, device=self.device) ## set uniform prior
        return prior


    def shape_prior(self, prior):
        if prior == None:
            prior = self.prior
        if list(prior.shape[-2:]) == [self.K, self.band]:
            prior = prior.float()  ## normal case
        elif prior.shape[-1] == self.band:
            prior = torch.repeat_interleave(prior.float().unsqueeze(-2), self.K, dim=-2)
        elif prior.shape[-1] == 1:
            prior = torch.repeat_interleave(prior.float(), self.K, dim=-1)
            prior = torch.repeat_interleave(prior.unsqueeze(-1), self.band, dim=-1)
        else:
            assert False, f"{self.prior_name} should be of shape {self.K, self.band}"
        return prior


##_________________________________________________________________________


def banded_diagonal(band, device = "cpu", epsilon=1e-4):
    """
    This creates a banded, square matrix of shape H x H from a smalled band matrix of shape H x b

    Parameters:
        -- band: a H x b dimensional torch vector
        -- all_simplex: roll over edges so that each row is simplex and sums to 1
    """

    H = band.shape[-2]
    b = band.shape[-1] - 1
    leading_dims = list(band.shape[:-2])

    mask = sum(torch.diagflat(torch.ones(H - abs(i), device=device), i) for i in range(-b // 2, b // 2 + 1))
    indeces = (mask == 1.0).nonzero(as_tuple=True)
    # band_1D = band.view(*band.shape[:-2], band.shape[-2] * band.shape[-1])
    band_1D = torch.flatten(band)[:band.shape[-2] * band.shape[-1]]
    mask[indeces] = band_1D[..., 1:-1]
    mask += epsilon

    mask = mask.repeat(leading_dims + [1, 1])
    return mask


if __name__ == '__main__':

    H = 3
    b = 3

    bmd = BMD(K=H, band=b)
    print(bmd.sample())

    band = torch.randint(0, 2, (H, b)).float()
    banded = banded_diagonal(band)
    dir_banded = dist.Dirichlet(banded).sample()
    print(dir_banded)

