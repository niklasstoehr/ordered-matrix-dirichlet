import torch
import pyro
import pyro.distributions as dist



class OMD:

    def __init__(self, K, A, priors={}, site_name="phi_ka", event_dim=2, device="cpu"):

        self.device = device
        self.K = K
        self.A = A
        self.site_name = site_name
        self.priors = priors
        self.event_dim = event_dim
        self.prior = self.select_prior(priors)
        self.dir_type = "omd"


    def sample(self, prior = None, n_samples = 1):
        prior = self.shape_prior(prior)
        if n_samples > 1:
            prior = torch.repeat_interleave(prior.unsqueeze(0), n_samples, dim=0)
        phi_a = prior
        phi_ka = pyro.deterministic(self.site_name, ordered_dirichlet(alpha=phi_a, K=self.K, site_name=self.site_name, device=self.device), event_dim=self.event_dim)
        return phi_ka

    def select_prior(self, prior):
        self.prior_name = self.site_name + "_omd"
        if self.prior_name in prior.keys():
            prior = self.shape_prior(prior[self.prior_name].to(self.device))
        else:  ## default prior
            prior = torch.ones(self.A, device=self.device) * 1.0 ## set uniform prior
        return prior

    def shape_prior(self, prior):
        if prior == None:
            prior = self.prior
        if prior.shape[-1] == self.A:
            prior = prior.float()  ## normal case
        elif prior.shape[-1] == 1:
            prior = torch.repeat_interleave(prior.float(), self.A, dim=-1)
        else:
            assert False, f"{self.prior_name} should be of shape {self.A}"
        return prior

##_________________________________________________________________________

def ordered_beta(a1, a2, K, A, site_name=None):
    a1 = torch.repeat_interleave(a1.unsqueeze(-1), K, dim=-1)
    a2 = torch.repeat_interleave(a2.unsqueeze(-1), K, dim=-1)
    vec = pyro.sample("{}_beta_{}".format(str(site_name), str(A)), dist.Beta(a1, a2).to_event(1))
    vec_sorted, vec_indices = vec.sort(descending=True)
    return vec_sorted[::1]


def ordered_dirichlet(alpha, K, site_name = None, device = "cpu"):
    """This generates an ordered set of K vectors x1 ... xK that are each
       of dimension A and sum to 1:

            \sum_{i=1}^A  x_{ki} = 1 for all k

       Furthermore, vector x_c is a discrete distribution
       which first-order stochastically dominates (FSD) x_{(k+1)}:
       which first-order stochastically dominates (FSD) x_{(k+1)}:

            \sum_{i=1}^m x_{ki} >= \sum_{i=1}^m x_{(k+1)i} for all m

        Parameters: 
            -- alpha: a A-dimensional array of shape parameters (mean * conc) pointwise
            -- size:  the number K of ordered vectors
            -- site_name: optional name of sample site for beta distribution
    """

    A = alpha.shape[-1]
    ordered_d0 = ordered_beta(a1=alpha[..., 0], a2=torch.sum(alpha[..., 1:], dim=-1), K=K, A=0, site_name=site_name)

    #X_KA = torch.zeros((*ordered_d0.shape[:-1], K, A))
    X_KA = torch.zeros((*ordered_d0.shape[:-1], K, A), device=device)
    X_KA[..., 0] = ordered_d0

    for d in range(1, A - 1):
        psi_C = ordered_beta(a1=alpha[..., d], a2=torch.sum(alpha[..., d+1:], dim=-1), K=K, A=d, site_name=site_name)
        X_KA[..., d] = (1 - torch.sum(X_KA[..., :d], dim=-1)) * psi_C
    X_KA[..., -1] = 1 - torch.sum(X_KA[..., :-1], dim=-1)

    ## clipping values that are smaller than zero
    #X_KA[X_KA < 0.0] = 0.0
    #norm = X_KA.norm(p=1, dim=-1, keepdim=True)
    #X_KA = X_KA.div(norm.expand_as(X_KA))
    return X_KA


if __name__ == '__main__':

    K = 10
    A = 20

    #alpha = dist.Dirichlet(torch.ones(A)).sample()
    alpha =torch.ones(A)
    X_KA = ordered_dirichlet(alpha, K= K)
    print(X_KA.shape)

    ordDir_site = OMD(K, A, site_name="phi_ka_omd", priors={}, event_dim=2)
    print("beta_a", ordDir_site.prior)
    print("sample", ordDir_site.sample())

