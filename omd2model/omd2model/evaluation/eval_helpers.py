import torch
from omd2model.handlers import tracing

def get_forecast_params(train_params, obs_sites=["x"], hidden_sites=["h", "delta_t"]):
    ### remove observed params and set initial hidden state
    forecast_params = dict()
    last_hidden_state = -1
    for k,v in train_params.items():
        k_split = k.split("/")
        if k_split[0] not in obs_sites and k_split[0] not in hidden_sites: ## remove observed states
            forecast_params[k] = v
        elif k_split[0] in hidden_sites: ## find last hidden state
            if last_hidden_state < int(k_split[1]):
                last_hidden_state = int(k_split[1])
    ## set initial hidden state observed states
    for hidden_site in hidden_sites:
        h = hidden_site + "/" + str(last_hidden_state)
        if h in train_params.keys():
            forecast_params[f"{hidden_site}/-1"] = train_params[h]
    return forecast_params


def predictive_trace(model, test_data, params_samples, infer=-1, combine_seq=True, obs_site="x"):
    pred_data = {"n_seq": test_data[obs_site]["value"].shape[1], "max_len": test_data[obs_site]["value"].shape[-1], obs_site: {"mask": test_data[obs_site]["mask"]}} ## copy over old mask from test_data
    pred_trace = tracing.trace_handler(model, pred_data, params_samples=params_samples, infer=infer, combine_seq=combine_seq)
    return pred_trace


def get_mean_mode(hat_value):
    if hat_value.shape[0] != 1:
        if hat_value.is_floating_point():
            hat_value = torch.mean(hat_value, dim=0).unsqueeze(0)
        else:
            hat_value = torch.mode(hat_value, dim=0)[0].unsqueeze(0)
    return hat_value


