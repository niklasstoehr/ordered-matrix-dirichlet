
import torch

import omd2model.evaluation.eval_helpers
from omd2model.models.pgds.pgds_model import PGDS
from omd2model.handlers import generation, inference, data_splitting, tracing
from omd2model.evaluation import eval_metrics, eval_helpers

## Data________

gt_prior = {"h/-1": torch.FloatTensor([1, 1, 1]),
            "emis_ka_bmd": torch.tensor(([[800, 1, 1], [1, 800, 1], [1, 1, 800]])),
            "emis_ka_smd": torch.tensor(([[800, 1, 1], [1, 800, 1], [1, 1, 800]])),
            "emis_ka_omd": torch.tensor(([10, 10, 10])),
            "trans_kk_bmd": torch.tensor(([[800, 1, 1], [1, 800, 1], [1, 1, 800]])),
            "trans_kk_smd": torch.tensor(([[100, 1, 1], [1, 100, 1], [1, 1, 100]])),
            "trans_kk_omd": torch.tensor(([10, 10, 10]))}

gt_pgds = PGDS(K=3, A=3, emis_type="smd", trans_type="smd", prior=gt_prior)
gt_values, gt_params, gt_trace = generation.generate_data(gt_pgds.model, {"n_seq": 20, "max_len": 10}, sites=["x", "h"])
train_data, test_data = data_splitting.train_test_split(gt_values["x"]["value"], test_type="impute", dim=[-1], frac=0.4)
#vizuals.trajectory_plot(train_data["x"]["value"][0][2].long(), title="train true – x", xlabel="time step",ylabel="counts", fig_size=(10, 2))

## Inference________

pgds = PGDS(K=3, A=3, emis_type = "smd", trans_type = "smd")
mcmc = inference.run_mcmc(pgds.model, train_data, num_samples=5, warmup_steps=5)
train_post_mean = inference.get_post_mean(mcmc.get_samples(), pgds.model, train_data, gt_params.keys())
forecast_params = eval_helpers.get_forecast_params(mcmc.get_samples())

## Evaluation________
fit_model = pgds.model

train_trace = tracing.trace_handler(fit_model, train_data, params_samples=mcmc.get_samples())
eval_metrics.evalute_post_dens(train_trace.nodes["x"]["log_prob"],mask=train_data["x"]["idx"],post_type="train post")

test_trace = tracing.trace_handler(fit_model, test_data, params_samples=mcmc.get_samples())
eval_metrics.evalute_post_dens(test_trace.nodes["x"]["log_prob"],mask=test_data["x"]["idx"],post_type="test post")

eval_metrics.evalute_post_dens(gt_trace.nodes["x"]["log_prob"], mask=test_data["x"]["gt_idx"], post_type="ground truth test")

## point predictions
pred_trace = eval_helpers.predictive_trace(fit_model, test_data, params_samples=mcmc.get_samples(), infer=-1)
eval_metrics.eval_point_pred(pred_trace.nodes["x"]["value"], gt_trace.nodes["x"]["value"], mask=test_data["x"]["idx"], gt_mask=test_data["x"]["gt_idx"], post_type="x")
eval_metrics.eval_point_pred(pred_trace.nodes["h"]["value"], gt_trace.nodes["h"]["value"], mask=None, post_type="h")
