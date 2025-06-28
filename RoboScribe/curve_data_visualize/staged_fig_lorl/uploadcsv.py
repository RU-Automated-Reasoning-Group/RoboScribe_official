# import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


fig, axs = plt.subplots(1, figsize=(9, 5))

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.set_title('Stage-wise progresses on MetaWorld task sequence')

# axs[1].spines['top'].set_visible(False)
# axs[1].spines['right'].set_visible(False)
# axs[1].set_title('T2R training progress on PickCube')

# axs_bl = axs[1]
# bl_grasp = "bl_pickcube_grasp.csv"
# bl_goal = "bl_pickcube_goal.csv"

# gs = "global_step"
# succ_mean = "Grouped runs - eval/cubeGrasp"
# succ_min = "Grouped runs - eval/cubeGrasp__MIN"
# succ_max = "Grouped runs - eval/cubeGrasp__MAX"
# grasp_df = pd.read_csv(bl_grasp)
# startfrom = 0

# grasp_step = grasp_df[gs]
# grasp_succ_mean = grasp_df[succ_mean]
# grasp_succ_min = grasp_df[succ_min]
# grasp_succ_max = grasp_df[succ_max]

# grasp_step_new = [0]
# grasp_succ_mean_new = [0]
# grasp_succ_min_new = [0]
# grasp_succ_max_new = [0]
# for i in range(len(grasp_step)):
#     if grasp_step[i]<=1040000:
#         grasp_step_new.append(grasp_step[i])
#         grasp_succ_mean_new.append(grasp_succ_mean[i]/2)
#         grasp_succ_min_new.append(grasp_succ_min[i]/2)
#         grasp_succ_max_new.append(grasp_succ_max[i]/2)
        
# axs_bl.plot(grasp_step_new, grasp_succ_mean_new,  color='#5B81D5')
# axs_bl.fill_between(x=grasp_step_new, y1=grasp_succ_min_new, y2=grasp_succ_max_new, color=('#5B81D5',0.3))


# gs = "global_step"
# succ_mean = "Grouped runs - eval/cubeGoal"
# succ_min = "Grouped runs - eval/cubeGoal__MIN"
# succ_max = "Grouped runs - eval/cubeGoal__MAX"
# goal_df = pd.read_csv(bl_goal)
# startfrom+=1040000

# goal_step = goal_df[gs]
# goal_succ_mean = goal_df[succ_mean]
# goal_succ_min = goal_df[succ_min]
# goal_succ_max = goal_df[succ_max]

# goal_step_new = [startfrom]
# goal_succ_mean_new = [0.5]
# goal_succ_min_new = [0.5]
# goal_succ_max_new = [0.5]
# for i in range(len(grasp_step)):
#     if goal_step[i]<=3e6 and goal_step[i]>=startfrom:
#         goal_step_new.append(goal_step[i])
#         goal_succ_mean_new.append((goal_succ_mean[i]+1)/2)
#         goal_succ_min_new.append((goal_succ_min[i]+1)/2)
#         goal_succ_max_new.append((goal_succ_max[i]+1)/2)
        
# axs_bl.plot(goal_step_new, goal_succ_mean_new,  color='#5B81D5', label='T2R')
# axs_bl.fill_between(x=goal_step_new, y1=goal_succ_min_new, y2=goal_succ_max_new, color=('#5B81D5',0.3))

# axs_bl.add_patch(plt.Rectangle((0, 0), 1/3, 1/2, transform=axs_bl.transAxes, color='royalblue', alpha=0.075))
# axs_bl.add_patch(plt.Rectangle((1/3, 1/2), 2/3, 1/2, transform=axs_bl.transAxes, color='navy', alpha=0.1))

# axs_bl.set_xlim(0,3e6)
# axs_bl.set_ylim(0,1)
# axs_bl.set_xlabel('Total timesteps')
# axs_bl.set_ylabel('Progress')
# axs_bl.legend(prop={'size': 8})

# axs_bl.set_yticks([0, 0.5, 1])
# # axs_bl.set_yticklabels(['$0$', r'$\frac{1}{2}$', '$1$'])
# axs_bl.tick_params(axis='y', rotation=45, labelsize=9)
# #axs_bl.set_yticklabels(['start', 'grasp\ncube', 'stack\ncube'])
# axs_bl.set_yticklabels(['Initiate', 'Grasp', 'Stack'])

l2s_mugback = "lorl_pushmugback2.csv"
l2s_opendrawer = "lorl_opendrawer3.csv"
l2s_faucetleft = "lorl_turnfaucetleft2.csv"

gs = "global_step"
succ_mean = "Grouped runs - rollout/success_rate"
succ_min = "Grouped runs - rollout/success_rate__MIN"
succ_max = "Grouped runs - rollout/success_rate__MAX"
grasp_df = pd.read_csv(l2s_mugback)
axs_l2s = axs


axs_l2s.plot([0,1e6], [0,0],  color='#C74A45')



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
        
axs_l2s.plot(grasp_step_new, grasp_succ_mean_new,  color='#C74A45')
axs_l2s.fill_between(x=grasp_step_new, y1=grasp_succ_min_new, y2=grasp_succ_max_new, color=('#C74A45', 0.3))


goal_df = pd.read_csv(l2s_opendrawer)


goal_step = goal_df[gs]
goal_succ_mean = goal_df[succ_mean]
goal_succ_min = goal_df[succ_min]
goal_succ_max = goal_df[succ_max]

startfrom+=1000000

goal_step_new = [startfrom]
goal_succ_mean_new = [0.33]
goal_succ_min_new = [0.33]
goal_succ_max_new = [0.33]


startfrom+=1000000
goal_step_new.append(startfrom)
goal_succ_mean_new.append(0.33)
goal_succ_min_new.append(0.33)
goal_succ_max_new.append(0.33)

for i in range(len(goal_step)):
    if goal_step[i]<=3e6 :
        goal_step_new.append(goal_step[i]+startfrom)
        goal_succ_mean_new.append((goal_succ_mean[i]+1)/3)
        goal_succ_min_new.append((goal_succ_min[i]+1)/3)
        goal_succ_max_new.append((goal_succ_max[i]+1)/3)


axs_l2s.plot(goal_step_new, goal_succ_mean_new,  color='#C74A45', label='L2S')
axs_l2s.fill_between(x=goal_step_new, y1=goal_succ_min_new, y2=goal_succ_max_new, color=('#C74A45',0.3))



goal_df = pd.read_csv(l2s_faucetleft)

goal_step = goal_df[gs]
goal_succ_mean = goal_df[succ_mean]
goal_succ_min = goal_df[succ_min]
goal_succ_max = goal_df[succ_max]

startfrom+=1000000

goal_step_new = [startfrom]
goal_succ_mean_new = [0.66]
goal_succ_min_new = [0.66]
goal_succ_max_new = [0.66]


startfrom+=1000000
goal_step_new.append(startfrom)
goal_succ_mean_new.append(0.66)
goal_succ_min_new.append(0.66)
goal_succ_max_new.append(0.66)

for i in range(len(goal_step)):
    if goal_step[i]<=4e6 :
        goal_step_new.append(goal_step[i]+startfrom)
        goal_succ_mean_new.append((goal_succ_mean[i]+2)/3)
        goal_succ_min_new.append((goal_succ_min[i]+2)/3)
        goal_succ_max_new.append((goal_succ_max[i]+2)/3)


axs_l2s.plot(goal_step_new, goal_succ_mean_new,  color='#C74A45', label='L2S')
axs_l2s.fill_between(x=goal_step_new, y1=goal_succ_min_new, y2=goal_succ_max_new, color=('#C74A45',0.3))


axs_l2s.legend(prop={'size': 8})
axs_l2s.set_xlim(0,6e6)
axs_l2s.set_ylim(0,1)
axs_l2s.set_xlabel('Total timesteps(steps)')
axs_l2s.set_ylabel('Progress(%)')


axs_l2s.add_patch(plt.Rectangle((0, 0), 1/3, 1/3, transform=axs_l2s.transAxes, color='darkred', alpha=0.1))
axs_l2s.add_patch(plt.Rectangle((1/3, 1/3), 1/3, 1/3, transform=axs_l2s.transAxes, color='darkred', alpha=0.1))
axs_l2s.add_patch(plt.Rectangle((2/3, 2/3), 1/3, 1/3, transform=axs_l2s.transAxes, color='darkred', alpha=0.1))

axs_l2s.set_yticks([0, 0.33,0.66, 1])
# axs_l2s.set_yticklabels(['$0$', r'$\frac{1}{2}$', '$1$'])
axs_l2s.tick_params(axis='y', rotation=45, labelsize=9)
axs_l2s.set_yticklabels(['Initiate', 'PushMugBack', 'OpenDrawer','Turn FaucetLeft'])

fig.tight_layout()
fig.savefig("test.pdf")