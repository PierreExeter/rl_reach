"""
Extract tuned hyperparameters after optimisation run
"""

import yaml


LOG_FOLDER = "../logs/opti/ppo/widowx_reacher-v1_1/"

with open(LOG_FOLDER + "default_hyperparams.yml", "r") as f:
    default_hyperparams = yaml.safe_load(f)

with open(LOG_FOLDER + "tuned_hyperparams.yml", "r") as f:
    d = yaml.safe_load(f)

print(default_hyperparams)


# for PPO only (do the same with the other algos)
if d['batch_size'] > d['n_steps']:
    d['batch_size'] = d['n_steps']

d['net_arch'] = {
    "small": "[dict(pi=[64, 64], vf=[64, 64])]",
    "medium": "[dict(pi=[256, 256], vf=[256, 256])]",
}[d['net_arch']]

d['activation_fn'] = {
    "tanh": "nn.Tanh",
    "relu": "nn.ReLU",
    "elu": "nn.ELU",
    "leaky_relu": "nn.LeakyReLU"}[
        d['activation_fn']]

d['policy_kwargs'] = "dict(log_std_init=" + str(d['log_std_init']) + ", net_arch=" + \
    d['net_arch'] + ", activation_fn=" + d['activation_fn'] + ", ortho_init=False)"

print(d)


# update defaults parameters
default_hyperparams['widowx_reacher-v1']['batch_size'] = d['batch_size']
default_hyperparams['widowx_reacher-v1']['clip_range'] = d['clip_range']
default_hyperparams['widowx_reacher-v1']['ent_coef'] = d['ent_coef']
default_hyperparams['widowx_reacher-v1']['gae_lambda'] = d['gae_lambda']
default_hyperparams['widowx_reacher-v1']['gamma'] = d['gamma']
default_hyperparams['widowx_reacher-v1']['learning_rate'] = d['lr']
default_hyperparams['widowx_reacher-v1']['max_grad_norm'] = d['max_grad_norm']
default_hyperparams['widowx_reacher-v1']['n_epochs'] = d['n_epochs']
default_hyperparams['widowx_reacher-v1']['n_steps'] = d['n_steps']
default_hyperparams['widowx_reacher-v1']['sde_sample_freq'] = d['sde_sample_freq']
default_hyperparams['widowx_reacher-v1']['vf_coef'] = d['vf_coef']
default_hyperparams['widowx_reacher-v1']['policy_kwargs'] = d['policy_kwargs']

with open(LOG_FOLDER + "final_hyperparams.yml", 'w') as f:
    yaml.dump(default_hyperparams, f)
