
import torch
import copy

from omd2model.models.lda import lda_model
from omd2model.handlers import generation, inference, data_splitting, tracing
from omd2model.evaluation import eval_metrics, eval_helpers

## Data________

C = 3
n_voc = 100

gt_prior = {"alpha_c": torch.randint(1,5,(1, C)).float(),
            "trans_kk_smd": torch.randint(1,5,(C, n_voc)).float()}

gt_lda = lda_model.LDA(n_c=C, n_voc=n_voc, phi_dir_type="smd", prior=gt_prior)
gt_values, gt_params, gt_trace = generation.generate_data(gt_lda.model, {"n_docs": 1000, "n_tokens": 10}, sites=["x"], infer=1, combine_seq=False)
train_data, test_data = data_splitting.train_test_split(gt_values["x"]["value"], test_type="impute", frac=0.3)
eval_metrics.evalute_post_dens(gt_trace.nodes["x"]["log_prob"], mask=test_data["x"]["mask"], post_type="ground truth")

## Inference________

lda = lda_model.LDA(n_c=C, n_voc=n_voc, phi_dir_type="smd")
mcmc = inference.run_mcmc(lda.model, train_data, num_samples=50, warmup_steps=10)
train_post_mean = inference.get_post_mean(mcmc.get_samples(), lda.model, train_data, gt_params.keys())

## Evaluation________

fit_model = lda.model
train_trace = tracing.trace_handler(fit_model, train_data, params_samples=mcmc.get_samples(), infer=0)
eval_metrics.evalute_post_dens(train_trace.nodes["x"]["log_prob"],mask=train_data["x"]["idx"],post_type="train post")
test_trace = tracing.trace_handler(fit_model, test_data, params_samples=mcmc.get_samples(), infer=0)
eval_metrics.evalute_post_dens(test_trace.nodes["x"]["log_prob"],mask=test_data["x"]["idx"],post_type="test post")

