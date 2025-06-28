
import numpy as np
import os
import pickle
import copy

import pdb

DEBUG = False

def get_move_action(observation, target_position, gain=10., close_gripper=False, block_id=None):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    # current_position = observation['observation'][:3]
    if block_id is None:
        current_position = observation[:3]
    else:
        current_position = observation[5+block_id*3: 5+block_id*3+3]

    action = gain * np.subtract(target_position, current_position)
    if close_gripper:
        gripper_action = -1.
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action

def block_is_grasped(obs, relative_grasp_position, block_position, dist_atol=3e-2, other_atol=1e-3):
    return block_inside_grippers(obs, relative_grasp_position, block_position, atol=dist_atol) and grippers_are_closed(obs, atol=other_atol)

def block_inside_grippers(obs, relative_grasp_position, block_position, atol=3e-2):
    # gripper_position = obs['observation'][:3]
    gripper_position = obs[:3]
    relative_position = np.subtract(gripper_position, block_position)

    return np.linalg.norm(relative_position - relative_grasp_position) < atol

# def grippers_are_closed(obs, atol=1e-3):
#     gripper_state = obs['observation'][3:5]
#     return abs(gripper_state[0] - 0.024) < atol

# def grippers_are_open(obs, atol=1e-3):
#     gripper_state = obs['observation'][3:5]
#     return abs(gripper_state[0] - 0.05) < atol

def grippers_are_closed(obs, atol=1e-3):
    threshold = 0.026
    gripper_state = obs[3:5]
    return (
        abs(np.sum(gripper_state) - 2*threshold) < atol or np.sum(gripper_state) - 2*threshold < 0
    )

def grippers_are_open(obs, atol=1e-3):
    threshold = 0.026
    gripper_state = obs[3:5]
    # return abs(gripper_state[0] - 0.05) < atol
    return np.sum(gripper_state) > 2 * threshold + atol

def get_custom_block_control(obs, all_block_ids, relative_grasp_position=(0., 0., -0.02), workspace_height=0.08, max_height=0.585, \
                             dist_atol=3e-2, goal_atol=3e-2, other_atol=1e-3, gain=10, block_id=0, last_block=False):
    """
    Returns
    -------
    action : [float] * 4
    """
    # find current goal
    sep_dim = obs.shape[0] // 2
    all_success = True
    for check_block in all_block_ids:
        cur_block = obs[5+check_block*3: 5+check_block*3+3]
        cur_goal = obs[sep_dim+5+check_block*3: sep_dim+5+check_block*3+3]
        if np.linalg.norm(cur_block - cur_goal) >= goal_atol or np.linalg.norm(cur_block[:2] - cur_goal[:2]) >= goal_atol:
            all_success = False
            break

    # gripper move
    if all_success:
        target_position = copy.deepcopy(obs[sep_dim: sep_dim+3])
        if DEBUG:
            print("move gripper to goal")
        if np.linalg.norm(obs[:3] - target_position) < dist_atol:
            return np.array([0., 0., 0., -1.]), True
        
        return get_move_action(obs, target_position, close_gripper=True, gain=gain), False

    # get position
    gripper_position = obs[:3]
    block_position = obs[5+block_id*3: 5+block_id*3+3]
    place_position = obs[sep_dim+5+block_id*3: sep_dim+5+block_id*3+3]
    
    # If the block is already at the place position, do nothing except keep the gripper closed
    if np.linalg.norm(block_position - place_position) < goal_atol:
        # open and move up
        if not grippers_are_open(obs, atol=other_atol):
            if DEBUG:
                print("Open the grippers To Leave")
            return np.array([0., 0., 0., .2]), False
        
        # move up to leave block
        # target_position = np.add(block_position, relative_grasp_position)
        target_position = copy.deepcopy(block_position)
        if last_block:
            target_position[2] += workspace_height * 2    
        else:
            target_position[2] += workspace_height
        if gripper_position[2] - target_position[2] < 0:
            if DEBUG:
                # print('{},  {}'.format(gripper_position, target_position))
                print("Move Up to Leave")
            # return get_move_action(obs, target_position, gain=gain * 0.1), False
            return np.array([0., 0., 0.5, 0.]), False

        if DEBUG:
            print("The block is already at the place position; do nothing")

        return np.array([0., 0., 0., 0.]), True

    # If gripper is already above place position
    block_is_grasped_check = block_is_grasped(obs, relative_grasp_position, block_position, dist_atol=dist_atol, other_atol=other_atol)
    target_position = place_position
    if block_is_grasped_check and np.linalg.norm(block_position[:2] - target_position[:2]) < goal_atol:
        if DEBUG:
            print("Move down to the place position")
        action = get_move_action(obs, target_position, close_gripper=True, gain=gain)
        # action[0] = 0
        # action[1] = 0
        return action, False

    # If block high enough, move to above place position
    if place_position[2] + workspace_height > max_height:
        high_enough = block_position[2] - (place_position[2] + max_height) / 2
    else:
        high_enough = block_position[2] - place_position[2] - workspace_height
    if block_is_grasped_check and high_enough > 0:
        target_position = copy.deepcopy(place_position)
        target_position[2] = block_position[2]
        if DEBUG:
            print("Move to above place position")
        # print(obs[:3])
        # print(block_position)
        # print(target_position)
        # return get_move_action(obs, target_position, close_gripper=True, gain=gain), False
        return get_move_action(obs, target_position, close_gripper=True, gain=gain*2, block_id=block_id), False

    # If the gripper is already grasping the block
    if block_is_grasped_check:
        # Move to the place position while keeping the gripper closed
        target_position = copy.deepcopy(block_position)
        target_position[2] = place_position[2] + workspace_height * 2
        if DEBUG:
            print("Move to above current position")
        return get_move_action(obs, target_position, close_gripper=True, gain=gain), False

    # If the block is ready to be grasped
    if block_inside_grippers(obs, relative_grasp_position, block_position, atol=dist_atol):

        # Close the grippers
        if DEBUG:
            print("Close the grippers")
        return np.array([0., 0., 0., -1.]), False

    # If the gripper is above the block
    if np.linalg.norm(gripper_position[:2] - block_position[:2]) < dist_atol/2:
    # if np.linalg.norm(gripper_position[:2] - block_position[:2]) < goal_atol:
        # If the grippers are closed, open them
        if not grippers_are_open(obs, atol=other_atol):
            if DEBUG:
                print("Open the grippers")
            return np.array([0., 0., 0., .01]), False

        # Move down to grasp
        target_position = np.add(block_position, relative_grasp_position)
        if DEBUG:
            print(obs[3:5])
            print("Move down to grasp")
        return get_move_action(obs, target_position, gain=gain), False

    # Else move the gripper to above the block
    target_position = np.add(block_position, relative_grasp_position)
    target_position[2] += workspace_height
    target_position[2] = min(target_position[2], max_height)
    target_position[2] = max(target_position[2], obs[2])
    if DEBUG:
        print("Move to above the block")
    return get_move_action(obs, target_position, gain=gain), False


