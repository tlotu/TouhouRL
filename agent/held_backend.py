"""Conservative PPO continuation for frame-duration-aware held decisions."""
from stable_baselines3.common.utils import FloatSchedule

from .ppo_backend import PPOBackend


class HeldPPOBackend(PPOBackend):
    def configure(self, config):
        self.held_config = config
        model = self.model
        model.learning_rate = config["learning_rate"]
        model.lr_schedule = FloatSchedule(model.learning_rate)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = model.learning_rate
        model.n_epochs = config["epochs"]
        model.batch_size = config["batch_size"]
        model.ent_coef = config["entropy_coefficient"]
        model.clip_range = FloatSchedule(config["clip_range"])
        model.target_kl = config["target_kl"]
        model.gamma = config["gamma_frame"]**config["hold_frames"]
        model.gae_lambda = config["lambda_frame"]**config["hold_frames"]

    def update(self, transitions, target_timesteps):
        c = self.held_config
        return super().update(transitions, target_timesteps,
                              frame_discounts=(c["gamma_frame"], c["lambda_frame"]))
