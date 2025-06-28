
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
        gripper_action = -1.
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action

def block_is_grasped(obs, relative_grasp_position, block_position, atol=1e-3):
    return block_inside_grippers(obs, relative_grasp_position, block_position, atol=atol) and grippers_are_closed(obs, atol=atol)

def block_inside_grippers(obs, relative_grasp_position, block_position, atol=1e-3):
    # gripper_position = obs['observation'][:3]
    gripper_position = obs[:3]
    relative_position = np.subtract(gripper_position, block_position)

    return np.sum(np.subtract(relative_position, relative_grasp_position)**2) < atol

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
        abs(gripper_state[0] - threshold) < atol or gripper_state[0] - threshold < 0
    )

def grippers_are_open(obs, atol=1e-3):
    threshold = 0.026
    gripper_state = obs[3:5]
    # return abs(gripper_state[0] - 0.05) < atol
    return gripper_state[0] > threshold + atol

def get_block_control(obs, relative_grasp_position=(0., 0., -0.02), workspace_height=0.1, atol=1e-3, gain=10, block_id=0, last_block=False):
    """
    Returns
    -------
    action : [float] * 4
    """
    # find current goal
    block_num = (obs.shape[0] - 13) // 18
    check_block = 0
    while check_block < block_num:
        cur_block = obs[10+check_block*15:10+check_block*15+3]
        cur_goal = obs[10+block_num*15+3*check_block:10+block_num*15+3*check_block+3]
        if np.sum(np.subtract(cur_block, cur_goal)**2) >= atol:
            break
        check_block += 1

    # only for now
    # if check_block < block_id:
    #     return None, False
    # elif check_block == block_num:
    #     return np.array([0., 0., 0., -1.]), True

    # get position
    gripper_position = obs[:3]
    block_position = obs[10+block_id*15:10+block_id*15+3]
    place_position = obs[10+block_num*15+3*block_id:10+block_num*15+3*block_id+3]
    
    # If the block is already at the place position, do nothing except keep the gripper closed
    if np.sum(np.subtract(block_position, place_position)**2) < atol:
        # open and move up
        if not grippers_are_open(obs, atol=atol):
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
            return get_move_action(obs, target_position, atol=atol, gain=gain), False

        if DEBUG:
            print("The block is already at the place position; do nothing")

        return np.array([0., 0., 0., 1.]), True

    # If gripper is already above place position
    target_position = place_position
    if (block_position[0] - target_position[0]) ** 2 + (block_position[1] - target_position[1]) ** 2 < atol:
        if DEBUG:
            print("Move down to the place position")
        return get_move_action(obs, target_position, atol=atol, close_gripper=True, gain=gain), False

    # If block high enough, move to above place position
    if (block_position[2] - place_position[2] - workspace_height) ** 2 < atol:
        target_position = copy.deepcopy(place_position)
        target_position[2] += workspace_height
        if DEBUG:
            print("Move to above place position")
        return get_move_action(obs, target_position, atol=atol, close_gripper=True, gain=gain), False

    # If the gripper is already grasping the block
    if block_is_grasped(obs, relative_grasp_position, block_position, atol=atol):
        # Move to the place position while keeping the gripper closed
        target_position = copy.deepcopy(block_position)
        target_position[2] = place_position[2] + workspace_height
        if DEBUG:
            print("Move to above current position")
        return get_move_action(obs, target_position, atol=atol, close_gripper=True, gain=gain), False

    # If the block is ready to be grasped
    if block_inside_grippers(obs, relative_grasp_position, block_position, atol=atol):

        # Close the grippers
        if DEBUG:
            print("Close the grippers")
        return np.array([0., 0., 0., -1.]), False

    # If the gripper is above the block
    if (gripper_position[0] - block_position[0])**2 + (gripper_position[1] - block_position[1])**2 < atol:

        # If the grippers are closed, open them
        if not grippers_are_open(obs, atol=atol):
            if DEBUG:
                print("Open the grippers")
            return np.array([0., 0., 0., 1.]), False

        # Move down to grasp
        target_position = np.add(block_position, relative_grasp_position)
        if DEBUG:
            print("Move down to grasp")
        return get_move_action(obs, target_position, atol=atol, gain=gain), False

    # Else move the gripper to above the block
    target_position = np.add(block_position, relative_grasp_position)
    target_position[2] += workspace_height
    if DEBUG:
        print("Move to above the block")
    return get_move_action(obs, target_position, atol=atol, gain=gain), False


