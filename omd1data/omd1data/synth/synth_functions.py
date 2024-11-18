
import torch
from torch.distributions.gamma import Gamma


def range_normalize(x, upper_bound=1):
    upper_bound = (upper_bound - 1)
    return (0 + (x - torch.min(x)) * (upper_bound - 0)) / (torch.max(x) - torch.min(x))


def option_picker(kwargs, options):
    if "dir" in kwargs:
        return kwargs["dir"]
    return options[torch.randint(0, len(options), (1,))]


def function(A, T, fn="linear", kwargs={}):
    def normal(A, T, kwargs):
        y = range_normalize(torch.rand((T)), A)
        return y

    def gamma(A, T, kwargs):
        gamma_dist = Gamma(torch.ones(T), torch.ones(T))
        y = range_normalize(gamma_dist.sample(), A)
        return y

    def constant(A, T, kwargs):
        c = torch.randint(1, A, (1,))
        y = c * torch.ones(T)
        return y

    def linear(A, T, kwargs):

        opt = option_picker(kwargs, ["up", "down"])
        if opt == "up":
            m = A / T
            c = 0
        if opt == "down":
            m = -(A / T)
            c = A - 1
        y = (m * torch.arange(0, T)) + c
        return y

    def sinus(A, T, kwargs):

        a = torch.arange(0, T)
        opt = option_picker(kwargs, ["up", "down"])
        if opt == "up":
            m = torch.sin(a)
        if opt == "down":
            m = -torch.sin(a)
        y = range_normalize(m, A)
        return y

    fn_list = {
        'normal': normal,
        'gamma': gamma,
        'constant': constant,
        'linear': linear,
        'sinus': sinus,
    }

    return fn_list[fn](A, T, kwargs)



if __name__ == '__main__':

    A = 20
    T = 10
    y = function(A, T, fn="normal")
    print(y)