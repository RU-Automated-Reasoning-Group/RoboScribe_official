from typing import Optional

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from policy.rce_sb3.models import ContinuousCriticCls

class RCEPolicy(SACPolicy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # init for SAC
        super().__init__(
            *args, **kwargs
        )

    # rewrite critic network
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCriticCls:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCriticCls(**critic_kwargs).to(self.device)



MlpPolicy = RCEPolicy
CNNPolicy = None
MultiInputPolicy = None