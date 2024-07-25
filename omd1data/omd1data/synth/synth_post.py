import torch


## post_processing
def make_poisson(y, kwargs):
    if "poisson" in kwargs.keys():
        y_rate = y * kwargs["poisson"]  # rate parameter between 0 and 5
        y = torch.poisson(y_rate)
    return y

def normalize_time_series(y, kwargs):
    if "total_count" in kwargs.keys():
        y = (y / (y.sum(0) + 0.0001)) * kwargs["total_count"]  ## normalize by N_V
    return y

def add_random_noise(y, kwargs, min_y = None, max_y = None):
    if "noise" in kwargs.keys():
        if min_y == None:
            min_y = 0
        if max_y == None:
            max_y = torch.max(y)
        y_noise = torch.randn(y.shape) * (kwargs["noise"]/max_y)
        y = torch.clamp(y + y_noise, min=min_y, max=max_y)
    return y


if __name__ == '__main__':

    T = 10
    cnt_y = torch.rand((T))
    cnt_kwargs = {"poisson": 1, "total_count": 100, "noise": 10}

    cnt_y = make_poisson(y=cnt_y, kwargs=cnt_kwargs)
    cnt_y = normalize_time_series(y=cnt_y, kwargs=cnt_kwargs)
    cnt_y = add_random_noise(cnt_y, cnt_kwargs)
    print(cnt_y)