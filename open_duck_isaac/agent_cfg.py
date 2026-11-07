"""PPO (rsl_rl) settings for the MPC-residual walking policy."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class OpenDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
  num_steps_per_env = 24
  max_iterations = 3000
  save_interval = 25  # about every minute: little is lost if the GPU driver resets
  experiment_name = "open_duck_mpc_residual"
  empirical_normalization = False
  obs_groups = {"actor": ["policy"], "critic": ["critic"]}
  actor = RslRlMLPModelCfg(
    hidden_dims=[256, 128, 128],
    activation="elu",
    obs_normalization=True,
    # Small start noise: the MPC already walks, the residual starts near zero.
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
  )
  critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 128], activation="elu", obs_normalization=True)
  algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.003,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
  )
