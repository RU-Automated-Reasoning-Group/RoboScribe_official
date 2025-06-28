from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import Schedule

class SACLagPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # init for SAC
        super().__init__(
            *args, **kwargs
        )

    # rewrite build
    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        # critic for lagrangian
        if self.share_features_extractor:
            self.critic_cost = self.make_critic(features_extractor=self.actor.features_extractor)
            critic_cost_parameters = [param for name, param in self.critic_cost.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic_cost = self.make_critic(features_extractor=None)
            critic_cost_parameters = list(self.critic_cost.parameters())

        # critic target for lagrangian
        self.critic_cost_target = self.make_critic(features_extractor=None)
        self.critic_cost_target.load_state_dict(self.critic_cost.state_dict())

        self.critic_cost.optimizer = self.optimizer_class(
            critic_cost_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_cost_target.set_training_mode(False)

    # rewrite to inclue new critic
    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)

        # add
        self.critic_cost.set_training_mode(mode)


MlpPolicy = SACLagPolicy
CNNPolicy = None
MultiInputPolicy = None