from omd1data.conflict import count_tensor, icews_data
from omd2model.handlers import data_splitting

def compute_V_prior(data, norm_by=[-1, -2]):
    if len(data.shape) == 5:
        V_prior = data.mean(0).sum(-1).sum(-1).sum(-1)
    elif len(data.shape) == 4:
        V_prior = data.mean(0).sum(-1).sum(-1)
    for dim in norm_by:
        V_prior = (V_prior+1) / (data.shape[dim])
    V_prior = V_prior.squeeze()
    return V_prior


def compute_T_prior(data, norm_by=[]):
    if len(data.shape) == 5:
        T_prior = data.mean(0).sum(0).sum(0).sum(0) / (data.shape[1] * data.shape[2])
    elif len(data.shape) == 4:
        T_prior = data.mean(0).sum(0).sum(0) / (data.shape[1])
    for dim in norm_by:
        T_prior = (T_prior+1) / (data.shape[dim])
    return T_prior


def compute_A_prior(data, norm_by=[]):
    if len(data.shape) == 5:
        A_prior = data.mean(0).sum(0).sum(0).sum(-1) / (data.shape[1] * data.shape[2])
    elif len(data.shape) == 4:
        A_prior = data.mean(0).sum(0).sum(-1) / (data.shape[1])
    for dim in norm_by:
        A_prior = (A_prior+1) / (data.shape[dim])
    return A_prior


def prep_train_test(cb, test_type="impute", dim=[-1], frac=0.3):
    train, test = data_splitting.train_test_split(cb.data.unsqueeze(0), test_type=test_type, dim=dim, frac=frac)
    train_prior = {"V": compute_V_prior(train["x"]["value"]), "T": compute_T_prior(train["x"]["value"]), "A": compute_A_prior(train["x"]["value"])}
    test_prior = {"V": compute_V_prior(test["x"]["value"]), "T": compute_T_prior(test["x"]["value"]), "A": compute_A_prior(train["x"]["value"])}
    return train, train_prior, test, test_prior


if __name__ == '__main__':
    icews = icews_data.ICEWS(file_list=[2015,2016,2017,2018,2019,2020], V1=[], V2=[], start_end=["2000-01-01", "2022-01-01"])
    cb = count_tensor.CountBuilder(icews.df, time_level="M", action_level="G_A_rank", min_V_count=100)
    train, train_prior, test, test_prior = prep_train_test(cb, test_type="split", dim=[-1], frac=0.3)
    print(cb.data.shape)
    print(train["x"]["value"].shape, test["x"]["value"].shape)