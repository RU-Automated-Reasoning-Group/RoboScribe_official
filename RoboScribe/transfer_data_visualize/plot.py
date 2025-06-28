import pdb
import numpy  as np
import os
import matplotlib.pyplot as plt

def tower_baseline():
    # to single tower 4,5,6,7
    single_tower_4 = {4:[88, 95, 87], 
                      5:[0,3,1], 
                      6:[0,0,0],
                      7:[0,0,0]}
    single_tower_5 = {4:[92, 85, 86], 
                      5:[80, 89, 71], 
                      6:[41, 52, 45],
                      7:[1, 3, 2]}

    # to multi tower 4,5,6,7,8,9
    multi_tower_4 = {4:[82,82,84], 
                     5:[84,73,78], 
                     6:[47,55,51],
                     7:[26,38,26],
                     8:[20,28,21],
                     9:[6,13,6]}
    multi_tower_5 = {4:[81,68,74], 
                     5:[66,73,69], 
                     6:[30,52,43],
                     7:[23,47,33],
                     8:[40,13,24],
                     9:[8,20,8]}

    # to Pyramid 4,5,6,7,8,9
    pyramid_4 = {4:[72,53,61], 
                 5:[22,39,31], 
                 6:[8,17,10],
                 7:[2,8,6],
                 8:[2,2,2],
                 9:[0,0,0]}
    pyramid_5 = {4:[33,63,48], 
                 5:[11,30,24], 
                 6:[7,9,7],
                 7:[1,5,1],
                 8:[0,3,0],
                 9:[0,0,0]}

    return [[single_tower_4, single_tower_5], \
           [multi_tower_4, multi_tower_5], \
           [pyramid_4, pyramid_5]]

def tower_robo():
    # to single tower 4,5,6,7
    single_tower_4 = {4:[89,88,87], 
                      5:[56,56,54], 
                      6:[16,14,14],
                      7:[0, 0, 0]}
    single_tower_5 = {4:[93,91,90], 
                      5:[77,75,75], 
                      6:[49,47,47],
                      7:[16,12,12]}

    # to multi tower 4,5,6,7,8,9
    multi_tower_4 = {4:[90,87,88], 
                     5:[84,86,79], 
                     6:[72,76,68],
                     7:[61,60,58],
                     8:[49,42,43],
                     9:[33,27,25]}
    multi_tower_5 = {4:[89, 89, 88], 
                     5:[85,82,83], 
                     6:[72,72,71],
                     7:[61,65,64],
                     8:[50,49,49],
                     9:[39,34,32]}

    # to Pyramid 4,5,6,7,8,9
    pyramid_4 = {4:[91,90,90], 
                 5:[93,88,89], 
                 6:[82,80,79],
                 7:[63,50,47],
                 8:[47,40,39],
                 9:[33,33,26]}
    pyramid_5 = {4:[93, 91, 91], 
                 5:[90,89,89], 
                 6:[84,87,84],
                 7:[62,60,59],
                 8:[48,45,44],
                 9:[33,34,34]}

    return [[single_tower_4, single_tower_5], \
           [multi_tower_4, multi_tower_5], \
           [pyramid_4, pyramid_5]]

def tower_hist():
    # load data
    baseline_results = tower_baseline()
    robo_results = tower_robo()
    tabs = ['Tower ', 'Multi Towers ', 'Pyramid ']
    titles = ['Tower 4', 'Tower 5', 'Tower 4', 'Tower 5']
    super_titles = ['\nReNN', '\nRoboScribe']
    colors = ['r', 'cornflowerblue', 'g', 'purple', 'orange', 'peru']

    # init parameter
    width = 0.055
    offset = 0.06
    dist = 0.06
    size = [1.6, 2.3, 2.0]

    # draw histogram
    fig, ax = plt.subplots(1, 3, figsize=(20,4.8), gridspec_kw={'width_ratios':size})
    for fig_id in range(3):
        datas = [baseline_results[fig_id][0], baseline_results[fig_id][1], 
                 robo_results[fig_id][0], robo_results[fig_id][1]]
        # get label
        keys = sorted(list(datas[0].keys()))
        labels = []
        for k in keys:
            labels.append(tabs[fig_id] + str(k))
        ax[fig_id].set_facecolor('lavender')
        ax[fig_id].set_axisbelow(True)
        ax[fig_id].grid(axis='y', color='white', linewidth=1)
        ax[fig_id].set_ylim([0., 1.0])

        shift = 0
        title_xs = []
        for data_id, each_data in enumerate(datas):
            # get data
            # xs = np.arange(len(each_data)) + offset * len(each_data) * data_id + dist * data_id
            xs = np.arange(len(each_data)) * offset + shift
            shift = shift + offset * len(each_data) + dist
            title_xs.append((xs[0] + xs[-1])/2)

            ys = np.array([np.mean(each_data[k]) for k in keys]) / 100.0 + 0.02
            yerrs = [[-np.min(each_data[k])+np.mean(each_data[k]), np.max(each_data[k])-np.mean(each_data[k])] for k in keys]
            yerrs = np.transpose(np.array(yerrs)) / 100.0
            # print(each_data)
            print(xs)
            # print(ys)
            # print(yerrs)
            # draw
            if data_id == 0:
                for x, y, label, color in zip(xs, ys, labels, colors[:len(ys)]):
                    ax[fig_id].bar(x, y, width, color=color, label=label)    
                # ax[fig_id].bar(xs, ys, width, color=colors[:len(ys)], label=labels)
            else:
                ax[fig_id].bar(xs, ys, width, color=colors[:len(ys)])
            ax[fig_id].errorbar(xs, ys, yerrs, fmt='.', color='k', markersize=1, capsize=3)

        # set legend    
        super_title_xs = [(title_xs[2*i]+title_xs[2*i+1])/2 for i in range(len(title_xs)//2)]
        ax[fig_id].set_xticks(ticks=title_xs+super_title_xs, labels=titles+super_titles, fontsize='x-large')
        ax[fig_id].tick_params(axis='y', labelsize='large')
        for ele in ax[fig_id].get_xticklabels()[len(titles):]:
            ele.set_fontweight('bold')
        
        # ax[fig_id].set_xticks(ticks=super_title_xs, labels=super_titles, fontsize='large')
        if fig_id == 0:
            ax[fig_id].set_ylabel("Success Rate", fontsize='x-large')
        ax[fig_id].legend(fontsize="x-large")
        ax[fig_id].set_title(tabs[fig_id], fontsize="xx-large", fontweight='bold')

    ax[0].set_xlim([-0.1, size[0]])
    ax[1].set_xlim([-0.1, size[1]])
    ax[2].set_xlim([-0.1, size[1]])

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    # fig.tight_layout()
    fig.savefig("test.png")



if __name__ == '__main__':
    tower_hist()