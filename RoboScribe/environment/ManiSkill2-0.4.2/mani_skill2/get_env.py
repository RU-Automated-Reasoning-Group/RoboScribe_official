import mani_skill2.envs
import gym

def get_opendrawer():
    env = gym.make('OpenCabinetDrawer-v1', obs_mode="state", asset_root='environment/ManiSkill2-0.4.2/data/partnet_mobility/dataset')
    return env