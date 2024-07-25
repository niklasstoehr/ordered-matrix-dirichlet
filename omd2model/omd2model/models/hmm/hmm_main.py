
import torch

from omd2model.models.hmm import hmm_model
from omd2model.handlers import generation, inference, data_splitting, tracing
from omd2model.evaluation import eval_metrics, eval_helpers

## Data________

V = 20
T = 4
K = 3
A = 3

gt_prior = {"alpha_k": torch.randint(1,4,(1, K)).float(),
            "emis_ka_banddir": torch.tensor(([[800, 10, 10], [10, 800, 10], [10, 10, 800]])),
            "emis_ka_normdir": torch.tensor(([[800, 10, 10], [10, 800, 10], [10, 10, 800]])),
            "emis_ka_orddir": torch.tensor(([10, 10, 10])),
            "trans_kk_banddir": torch.tensor(([[800, 10, 10], [10, 800, 10], [10, 10, 800]])),
            "trans_kk_normdir": torch.tensor(([[800, 10, 10], [10, 800, 10], [10, 10, 800]])),
            "trans_kk_orddir": torch.tensor(([10, 10, 10]))}

gt_hmm = hmm_model.HMM(K=K, A=A, emis_type="smd", trans_type="smd", prior=gt_prior)
gt_values, gt_params, gt_trace = generation.generate_data(gt_hmm.model, {"n_seq": V, "max_len": T}, sites=["x", "h"], infer=-1)
train_data, test_data = data_splitting.train_test_split(gt_values["x"]["value"], test_type="impute", dim=[-1], frac=0.1)
eval_metrics.evalute_post_dens(gt_trace.nodes["x"]["log_prob"], mask=test_data["x"]["mask"], post_type="ground truth")

## Inference________

hmm = hmm_model.HMM(K=K, A=A, emis_type = "smd", trans_type = "smd")
mcmc = inference.run_mcmc(hmm.model, train_data, num_samples=5, warmup_steps=5)
train_post_mean = inference.get_post_mean(mcmc.get_samples(), hmm.model, train_data, gt_params.keys())


## Evaluation________
fit_model = hmm.model

train_trace = tracing.trace_handler(fit_model, train_data, params_samples=mcmc.get_samples(), infer=0)
eval_metrics.evalute_post_dens(train_trace.nodes["x"]["log_prob"],mask=train_data["x"]["idx"],post_type="train post")

test_trace = tracing.trace_handler(fit_model, test_data, params_samples=mcmc.get_samples(), infer=0)
eval_metrics.evalute_post_dens(test_trace.nodes["x"]["log_prob"],mask=test_data["x"]["idx"],post_type="test post")

eval_metrics.evalute_post_dens(gt_trace.nodes["x"]["log_prob"], mask=test_data["x"]["gt_idx"], post_type="ground truth test")

## point predictions
post_params = inference.infer_post_discrete(fit_model, train_data, mcmc.get_samples(), infer=0)
pred_trace = eval_helpers.predictive_trace(fit_model, test_data, params_samples=post_params, infer=-1)
eval_metrics.eval_point_pred(pred_trace.nodes["x"]["value"], gt_trace.nodes["x"]["value"], mask=test_data["x"]["idx"], gt_mask=test_data["x"]["gt_idx"], post_type="x")
eval_metrics.eval_point_pred(pred_trace.nodes["h"]["value"], gt_trace.nodes["h"]["value"], mask=None, post_type="h")