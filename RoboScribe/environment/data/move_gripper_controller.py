import numpy as np

def get_move_action(observation, target_position, atol=1e-3, gain=10., close_gripper=False):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    current_position = observation[:3]

    action = gain * np.subtract(target_position, current_position)
    if close_gripper:
        gripper_action = -1.
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action

class initGripper:
    def __init__(self, gripper_pos):
        self.init_gripper_pos = gripper_pos

    def train(self):
        pass

    def eval(self):
        pass

    def act(self, state, device):
        gripper_pos = state[:3]
        action = get_move_action(gripper_pos, self.init_gripper_pos)
        return action