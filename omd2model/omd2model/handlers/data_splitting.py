import numpy as np
import torch
import copy
DIAG_MASK = True

def train_test_split(values, test_type="split", dim=[-1], frac=0.2):

    values = values.float()

    ## split _________________________
    if test_type == "split":
        N = values.shape[dim[0]]
        split_N = int(N * (1 - frac))

        train_idx= torch.zeros(values.shape)
        train_ind = torch.arange(0, split_N, 1)
        for d in dim:
            train_idx = train_idx.index_fill(d, train_ind, 1).long()
        test_idx = (~train_idx.bool()).long()

        ## slice values
        train_values = torch.masked_select(values, train_idx.bool()).view(*train_idx.shape[:-1], -1)
        test_values = torch.masked_select(values, test_idx.bool()).view(*test_idx.shape[:-1], -1)

        ## add add diagonal mask
        train_values[diagonal_mask(torch.zeros(train_values.shape), mask_value=1).bool()] = np.nan ## set diag=1 to set nan next
        test_values[diagonal_mask(torch.zeros(test_values.shape), mask_value=1).bool()] = np.nan ## set diag=1 to set nan next

        train_data = set_value_mask(train_values, set_nan=0)
        test_data = set_value_mask(test_values, set_nan=0)

        ## add train_test split information
        train_data["x"]["gt_idx"] = diagonal_mask(train_idx, mask_value=0).long()  ## set diag=0 in index
        test_data["x"]["gt_idx"] = diagonal_mask(test_idx, mask_value=0).long()  ## set diag=0 in index

        train_data["x"]["idx"] = train_data["x"]["mask"].long()
        test_data["x"]["idx"] = test_data["x"]["mask"].long()


    ## impute unit__________________________
    if test_type == "impute":

        train_values, train_idx = remove_random(values, frac=frac)
        train_data = set_value_mask(train_values, set_nan=0)

        values[diagonal_mask(torch.zeros(values.shape), mask_value=1).bool()] = np.nan ## set diag=1 to set nan next
        test_data = set_value_mask(values, set_nan=0)  ## give full data as test, but pick out idx for evaluation

        ## add train_test split information
        test_idx = diagonal_mask(~train_idx, mask_value=0).bool()

        train_data["x"]["gt_idx"] = train_idx.long()
        test_data["x"]["gt_idx"] = test_idx.long()

        train_data["x"]["idx"] = train_data["x"]["gt_idx"]
        test_data["x"]["idx"] = test_data["x"]["gt_idx"]
    return train_data, test_data


def remove_random(values, frac=0.2):
    removed_values = copy.deepcopy(values).float()
    nan_mask = torch.bernoulli(torch.ones(removed_values.shape), p=frac).long()
    nan_mask = diagonal_mask(nan_mask, mask_value=1).bool()
    removed_values[nan_mask] = np.nan
    return removed_values, ~nan_mask ## ~ reverts to be index


def diagonal_mask(mask, mask_value=1):
    if DIAG_MASK and len(mask.shape) > 3:  ## get diagonal for V-V
        if mask.shape[0] == 1:  ## if dim = 0 is sample side, use dim = 1 for V
            eye = torch.eye(mask.shape[1]).unsqueeze(0) ## unsqueeze to add sample side
        else: ## if dim = 0 is V
            eye = torch.eye(mask.shape[0])
        for dim_i in range(1, int(len(mask.shape) - len(eye.shape) + 1)):
            eye = torch.repeat_interleave(eye.unsqueeze(-dim_i), mask.shape[-dim_i], dim=-dim_i)
        mask = mask.masked_fill(eye.bool(), mask_value).long()  ## missing values are 1 in mask
    return mask


def set_value_mask(values, site_name="x", set_nan=0):
    data = dict()
    mask = torch.where(torch.isnan(values), 1.0, 0.0).bool()  ## missing values are 1 in mask
    values[mask] = set_nan  ## fill nan values
    mask = ~mask ## important: pyro interprets 0 as False
    data[site_name] = {"value": values, "mask": mask}
    return data


if __name__ == '__main__':

    values = torch.rand(10,10,4,5)

    train, test = train_test_split(values, test_type="impute", dim=[-1], frac=0.5)
    print(train["x"]["value"].shape, test["x"]["value"].shape)
    print(train["x"]["mask"][0,0], train["x"]["mask"][0,1])
    print(train["x"]["mask"].sum())
    print(train["x"]["idx"].shape)

    train, test = train_test_split(values, test_type="split", dim=[-1], frac=0.5)
    print(train["x"]["value"].shape, test["x"]["value"].shape)
    print(train["x"]["mask"][0,0], train["x"]["mask"][0,1])
    print(train["x"]["mask"].sum())
    print(train["x"]["idx"][0,1])



