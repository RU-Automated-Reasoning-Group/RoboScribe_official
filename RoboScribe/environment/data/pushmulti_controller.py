import numpy as np
import copy
import pdb

DEBUG = False

# TODO: high enough -> push pos -> push

def get_pushmulti_control(obs, atol=1e-4, block_width=0.15, workspace_height=0.1, block_idx=0, goal_threshold=0.05, block_id=0, last_block=False):
    # only for debug
    # workspace_height = workspace_height
    atol = 1e-3

    # find current goal
    block_num = (obs.shape[0] - 10) // 19
    check_block = 0
    while check_block < block_num:
        cur_block = obs[10+check_block*16:10+check_block*16+3]
        cur_goal = obs[10+block_num*16+3*check_block:10+block_num*16+3*check_block+3]
        if np.sqrt(np.sum(np.subtract(cur_block, cur_goal)**2)) >= goal_threshold:
            break
        check_block += 1

    if check_block == block_num:
        return np.array([0., 0., 0., 0.]), True
    elif check_block < block_id:
        return None, False

    # get position
    gripper_position = obs[:3]
    block_position = obs[10+block_id*16:10+block_id*16+3]
    goal = obs[10+block_num*16+3*block_id:10+block_num*16+3*block_id+3]

    desired_block_angle = np.arctan2(goal[0] - block_position[0], goal[1] - block_position[1])
    gripper_angle = np.arctan2(goal[0] - gripper_position[0], goal[1] - gripper_position[1])

    push_position = block_position.copy()
    # push_position[0] += -1. * np.sin(desired_block_angle) * block_width / 2.
    # push_position[1] += -1. * np.cos(desired_block_angle) * block_width / 2.
    push_position[0] += -.8 * np.sin(desired_block_angle) * block_width / 2.
    push_position[1] += -.8 * np.cos(desired_block_angle) * block_width / 2.
    push_position[2] += 0.005

    # If the block is already at the place position, do nothing
    # if np.sum(np.subtract(block_position, goal)**2) < atol:
    if np.sqrt(np.sum(np.subtract(block_position, goal)**2)) <= goal_threshold:
        # move up to leave block
        target_position = copy.deepcopy(block_position)
        target_position[2] += workspace_height
        # if gripper_position[2] - target_position[2] < 0:
        if gripper_position[2] - block_position[2] - workspace_height*0.6 < 0:
            if DEBUG:
                print("Move Up to Leave")
            action = get_move_action(gripper_position, target_position, atol=atol)
            action[:1] = 0.
            return action, False
            # return np.array([0., 0., 0., 0.02]), True

        if DEBUG:
            print("The block is already at the place position; do nothing")
        return np.array([0., 0., 0., 0.]), True

    # Angle between gripper and goal vs block and goal is roughly the same
    angle_diff = abs((desired_block_angle - gripper_angle + np.pi) % (2*np.pi) - np.pi)

    gripper_sq_distance = (gripper_position[0] - goal[0])**2 + (gripper_position[1] - goal[1])**2
    block_sq_distance = (block_position[0] - goal[0])**2 + (block_position[1] - goal[1])**2

    # print('{}  {}  {}'.format((gripper_position[2] - push_position[2])**2, angle_diff, block_sq_distance-gripper_sq_distance))

    if (gripper_position[2] - push_position[2])**2 < atol and angle_diff < np.pi/4 and block_sq_distance < gripper_sq_distance:

        # Push towards the goal
        target_position = goal
        target_position[2] = gripper_position[2]
        if DEBUG:
            print("Push")
        return get_move_action(obs, target_position, atol=atol, gain=5.0), False

    # print(gripper_position)
    # print(push_position)
    # print((gripper_position[0] - push_position[0])**2 + (gripper_position[1] - push_position[1])**2)

    # If the gripper is above the push position
    if (gripper_position[0] - push_position[0])**2 + (gripper_position[1] - push_position[1])**2 < atol * 0.5:
    # if (gripper_position[0] - push_position[0])**2 + (gripper_position[1] - push_position[1])**2 < atol:

        # Move down to prepare for push
        if DEBUG:
            print("Move down to prepare for push")
        return get_move_action(obs, push_position, atol=atol), False

    # if block_id == 1:
    #     pdb.set_trace()

    if gripper_position[2] - push_position[2] - workspace_height * 0.5 > 0:
        # Else move the gripper to above the push
        target_position = push_position.copy()
        # target_position[2] += workspace_height
        target_position[2] = gripper_position[2]
        if DEBUG:
            print("Move to above the push position")
        action = get_move_action(obs, target_position, atol=atol)
        action[2] = 0.
        return action, False

    target_position = gripper_position.copy()
    target_position[2] = gripper_position[2] + workspace_height * 0.6
    if DEBUG:
        print("Move high enough")
    return get_move_action(obs, target_position, atol=atol), False



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