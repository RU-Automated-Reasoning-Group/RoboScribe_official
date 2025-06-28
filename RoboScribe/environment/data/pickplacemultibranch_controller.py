
import numpy as np
import os
import pickle
import copy

import pdb

DEBUG = False

def get_move_action(observation, target_position, atol=1e-3, gain=10., close_gripper=False):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    # current_position = observation['observation'][:3]
    current_position = observation[:3]

    action = gain * np.subtract(target_position, current_position)
    if close_gripper:
        # gripper_action = -1.
        gripper_action = -0.2
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action

def block_is_grasped(obs, relative_grasp_position, block_position, dist_atol=3e-2, other_atol=1e-3):
    return block_inside_grippers(obs, relative_grasp_position, block_position, atol=dist_atol) and grippers_are_closed(obs, atol=other_atol)

def block_inside_grippers(obs, relative_grasp_position, block_position, atol=1e-3):
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
    # return gripper_state[0] > threshold + atol
    return np.sum(gripper_state) > 2 * threshold + atol

def get_pickmultibranch_control(obs, relative_grasp_position=(0., 0., -0.02), workspace_height=0.1, dist_atol=3e-2, other_atol=1e-3, target_dist_atol=3e-2, gain=10, last_block=False, block_id=None, need_away=False):
    """
    Returns
    -------
    action : [float] * 4
    """
    # only two blocks for now
    block_num = 2
    assert need_away

    # cur_goal = obs[10+block_num*12 : 10+block_num*12+3]
    # if not last_block:
    #     cur_goal[-1] -= 0.05
    #     cur_block = obs[10+12:10+12+3]
    # else:
    #     cur_block = obs[10:13]
    cur_goal = obs[10+block_num*12+3*block_id : 10+block_num*12+3*block_id+3]
    cur_block = obs[10+12*block_id : 10+12*block_id+3]
    if last_block:
        assert cur_goal[2] > 0.43
    else:
        assert cur_goal[2] < 0.43

    # get position
    gripper_position = obs[:3]
    block_position = cur_block
    place_position = cur_goal
    
    # If the block is already at the place position, do nothing except keep the gripper closed
    if np.linalg.norm(block_position - place_position) < target_dist_atol:
        # open and move up
        if not grippers_are_open(obs, atol=other_atol):
            if DEBUG:
                print("Open the grippers To Leave")
            return np.array([0., 0., 0., 1.]), False
        
        # move up to leave block
        # target_position = np.add(block_position, relative_grasp_position)
        target_position = copy.deepcopy(block_position)
        if last_block:
            target_position[2] += workspace_height * 2    
        else:
            target_position[2] += workspace_height
        if gripper_position[2] - target_position[2] < 0:
            if DEBUG:
                print("Move Up to Leave")
            # return get_move_action(obs, target_position, atol=atol, gain=gain), False
            return np.array([0., 0., 0.5, 0.]), False

        if DEBUG:
            print("The block is already at the place position; do nothing")

        return np.array([0., 0., 0., .2]), True

    # If gripper is already above place position
    target_position = place_position
    if np.linalg.norm(block_position[:2] - target_position[:2]) < target_dist_atol:
        if DEBUG:
            print("Move down to the place position")
        return get_move_action(obs, target_position, atol=dist_atol, close_gripper=False, gain=gain), False

    # If block high enough, move to above place position
    # if place_position[2] > 0.6:
    #     used_workspace_height = 0
    # else:
    #     used_workspace_height = workspace_height
    used_workspace_height = workspace_height
    if block_is_grasped(obs, relative_grasp_position, block_position, dist_atol=dist_atol, other_atol=other_atol) and \
        (block_position[2] - place_position[2] - used_workspace_height) > 0:
        target_position = copy.deepcopy(place_position)
        target_position[2] += used_workspace_height
        if DEBUG:
            print("Move to above place position")
        return get_move_action(obs, target_position, atol=dist_atol, close_gripper=True, gain=gain), False

    # If the gripper is already grasping the block
    if block_is_grasped(obs, relative_grasp_position, block_position, dist_atol=dist_atol, other_atol=other_atol):
        # Move to the place position while keeping the gripper closed
        target_position = copy.deepcopy(block_position)
        target_position[2] = place_position[2] + used_workspace_height
        if DEBUG:
            print("Move to above current position")
        # return get_move_action(obs, target_position, atol=dist_atol, close_gripper=True, gain=gain), False
        return np.array([0., 0., 1., -0.2]), False

    # If the block is ready to be grasped
    if block_inside_grippers(obs, relative_grasp_position, block_position, atol=dist_atol):

        # Close the grippers
        if DEBUG:
            print("Close the grippers")
        return np.array([0., 0., 0., -1.]), False

    # If the gripper is above the block
    if np.linalg.norm(gripper_position[:2] - block_position[:2]) < dist_atol:

        # If the grippers are closed, open them
        if not grippers_are_open(obs, atol=other_atol):
            if DEBUG:
                print("Open the grippers")
            return np.array([0., 0., 0., .2]), False

        # Move down to grasp
        target_position = np.add(block_position, relative_grasp_position)
        if DEBUG:
            print("Move down to grasp")
        return get_move_action(obs, target_position, atol=dist_atol, gain=gain), False

    # open the gripper first
    if not grippers_are_open(obs, atol=other_atol):
        # Open the grippers
        if DEBUG:
            print("Open the grippers")
        return np.array([0., 0., 0., .2]), False

    # Else move the gripper to above the block
    if place_position[2] > block_position[2]:
        workspace_height = workspace_height * 2
    target_position = np.add(block_position, relative_grasp_position)
    target_position[2] += workspace_height
    if DEBUG:
        print("Move to above the block")
    return get_move_action(obs, target_position, atol=dist_atol, gain=gain), False


