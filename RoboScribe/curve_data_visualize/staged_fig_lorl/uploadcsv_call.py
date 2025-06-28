# import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


def do_plot(fig, axs, color):
    # fig, axs = plt.subplots(1, figsize=(9, 5))

    # axs.spines['top'].set_visible(False)
    # axs.spines['right'].set_visible(False)
    axs.set_title('Stage-wise progresses on MetaWorld task sequence')

    l2s_mugback = "staged_fig_lorl/lorl_pushmugback2.csv"
    l2s_opendrawer = "staged_fig_lorl/lorl_opendrawer3.csv"
    l2s_faucetleft = "staged_fig_lorl/lorl_turnfaucetleft2.csv"

    gs = "global_step"
    succ_mean = "Grouped runs - rollout/success_rate"
    succ_min = "Grouped runs - rollout/success_rate__MIN"
    succ_max = "Grouped runs - rollout/success_rate__MAX"
    grasp_df = pd.read_csv(l2s_mugback)
    axs_l2s = axs


    axs_l2s.plot([0,1e6], [0,0],  color=color)



    grasp_step = grasp_df[gs]
    grasp_succ_mean = grasp_df[succ_mean]
    grasp_succ_min = grasp_df[succ_min]
    grasp_succ_max = grasp_df[succ_max]

    grasp_step_new = [0]
    grasp_succ_mean_new = [0]
    grasp_succ_min_new = [0]
    grasp_succ_max_new = [0]

    startfrom = 1000000
    grasp_step_new.append(startfrom)
    grasp_succ_mean_new.append(0)
    grasp_succ_min_new.append(0)
    grasp_succ_max_new.append(0)


    for i in range(len(grasp_step)):
        if grasp_step[i]<=1000000 :
            grasp_step_new.append(grasp_step[i]+startfrom)
            grasp_succ_mean_new.append(grasp_succ_mean[i]/3)
            grasp_succ_min_new.append(grasp_succ_min[i]/3)
            grasp_succ_max_new.append(grasp_succ_max[i]/3)
            # grasp_succ_mean_new.append(grasp_succ_mean[i])
            # grasp_succ_min_new.append(grasp_succ_min[i])
            # grasp_succ_max_new.append(grasp_succ_max[i])
            
    axs_l2s.plot(grasp_step_new, grasp_succ_mean_new,  color=color)
    axs_l2s.fill_between(x=grasp_step_new, y1=grasp_succ_min_new, y2=grasp_succ_max_new, color=(color, 0.3))


    goal_df = pd.read_csv(l2s_opendrawer)


    goal_step = goal_df[gs]
    goal_succ_mean = goal_df[succ_mean]
    goal_succ_min = goal_df[succ_min]
    goal_succ_max = goal_df[succ_max]

    startfrom+=1000000

    goal_step_new = [startfrom]
    goal_succ_mean_new = [1/3]
    goal_succ_min_new = [1/3]
    goal_succ_max_new = [1/3]


    startfrom+=1000000
    goal_step_new.append(startfrom)
    goal_succ_mean_new.append(1/3)
    goal_succ_min_new.append(1/3)
    goal_succ_max_new.append(1/3)

    for i in range(len(goal_step)):
        if goal_step[i]<=3e6 :
            goal_step_new.append(goal_step[i]+startfrom)
            goal_succ_mean_new.append((goal_succ_mean[i]+1)/3)
            goal_succ_min_new.append((goal_succ_min[i]+1)/3)
            goal_succ_max_new.append((goal_succ_max[i]+1)/3)


    axs_l2s.plot(goal_step_new, goal_succ_mean_new,  color=color)
    axs_l2s.fill_between(x=goal_step_new, y1=goal_succ_min_new, y2=goal_succ_max_new, color=(color,0.3))



    goal_df = pd.read_csv(l2s_faucetleft)

    goal_step = goal_df[gs]
    goal_succ_mean = goal_df[succ_mean]
    goal_succ_min = goal_df[succ_min]
    goal_succ_max = goal_df[succ_max]

    startfrom+=1000000

    goal_step_new = [startfrom]
    goal_succ_mean_new = [2/3]
    goal_succ_min_new = [2/3]
    goal_succ_max_new = [2/3]


    startfrom+=1000000
    goal_step_new.append(startfrom)
    goal_succ_mean_new.append(2/3)
    goal_succ_min_new.append(2/3)
    goal_succ_max_new.append(2/3)

    for i in range(len(goal_step)):
        if goal_step[i]<=4e6 :
            goal_step_new.append(goal_step[i]+startfrom)
            goal_succ_mean_new.append((goal_succ_mean[i]+2)/3)
            goal_succ_min_new.append((goal_succ_min[i]+2)/3)
            goal_succ_max_new.append((goal_succ_max[i]+2)/3)


    axs_l2s.plot(goal_step_new, goal_succ_mean_new,  color=color)
    axs_l2s.fill_between(x=goal_step_new, y1=goal_succ_min_new, y2=goal_succ_max_new, color=(color,0.3))


    # axs_l2s.legend(prop={'size': 8})
    # axs_l2s.set_xlim(0,6e6)
    # axs_l2s.set_ylim(0,1)
    # axs_l2s.set_xlabel('Total timesteps(steps)')
    # axs_l2s.set_ylabel('Progress(%)')


    axs_l2s.add_patch(plt.Rectangle((0, 0+0.02), 1/3, 1/3, transform=axs_l2s.transAxes, color='darkred', alpha=0.1))
    axs_l2s.add_patch(plt.Rectangle((1/3, 1/3+0.02), 1/3, 1/3, transform=axs_l2s.transAxes, color='darkred', alpha=0.1))
    axs_l2s.add_patch(plt.Rectangle((2/3, 2/3+0.02), 1/3, 1/3, transform=axs_l2s.transAxes, color='darkred', alpha=0.1))

    axs_l2s.set_yticks([0, 1/3, 2/3, 1])
    # axs_l2s.tick_params(axis='y', rotation=45, labelsize=9)
    # axs_l2s.set_yticklabels(['Initiate', 'PushMugBack', 'OpenDrawer','Turn FaucetLeft'])
    axs_l2s.set_yticklabels(['', 't0', 't1','t2'], fontsize='medium')