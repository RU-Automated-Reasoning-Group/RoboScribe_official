from csv import DictReader
import pdb
import numpy  as np
import os
import matplotlib.pyplot as plt
from staged_fig_lorl.uploadcsv_call import do_plot

def parse_wandb_csv(fname):
    reader = DictReader(open(fname))
    xs, ys = [], []
    for row in reader:
        if "Step" not in row:
            pdb.set_trace()
        xs.append(float(row["Step"]))
        ys.append(float(row["Value"]))
    return xs, ys

def parse_goalgail_csv(fname, multipler=25000):
    reader = DictReader(open(fname))
    xs, ys = [], []
    for row in reader:
        xs.append(multipler * int(row["Outer_iter"]))
        ys.append(float(row["Outer_Success"]))
    return xs, ys

def parse_csv_with_leading_zero(fname, leading_0s):
    reader = DictReader(open(fname))
    xs, ys = [0, leading_0s], [0, 0]
    for row in reader:
        if "Step" not in row:
            pdb.set_trace()
        xs.append(float(row["Step"]) + leading_0s)
        ys.append(float(row["Value"]))
        # print(row['Step'])
    # import pdb; pdb.set_trace()
    return xs, ys

def parse_npy_with_leading_zero(fname, leading_0s):
    ys, xs = np.load(fname)
    if leading_0s != 0:
        xs = [0, leading_0s] + (xs+leading_0s).tolist()[::-1]
        ys = [0, 0] + ys.tolist()[::-1]
    else:
        xs = (xs+leading_0s).tolist()[::-1]
        ys = ys.tolist()[::-1]
    return xs, ys

def generate_plot_data(list_xs, list_ys):
    all_xs = set()
    for xs in list_xs:
        for x in xs:
            if x not in all_xs:
                all_xs.add(x)

    all_xs_sorted = sorted(list(all_xs))
    all_ys = []
    for xs, ys in zip(list_xs, list_ys):
        all_ys.append(np.interp(all_xs_sorted, xs, ys))

    all_ys = np.array(all_ys)
    mean_ys = np.mean(all_ys, axis=0)

    std_ys = np.std(all_ys, axis=0)
    return np.array(all_xs_sorted), mean_ys, std_ys         

def smooth(xs, ys, k=3):
    new_xs = xs[:2]
    new_ys = ys[:2]
    for i in range(k*2, len(xs), k):
        new_xs.append(xs[i])
        # print(i)
        new_ys.append(max(ys[i-k:i+1]))
    return new_xs, new_ys

def generate_leading_zero_data(dir, leading_zero, divide=True, type='csv', ratio=1.0, repeat_x=None, start_x=None, y_fun=None):
    wandb_files = os.listdir(dir)
    list_xs = []
    list_ys = []
    for f in wandb_files:
        if type == 'csv':
            xs, ys = parse_csv_with_leading_zero(os.path.join(dir, f), leading_zero)
        elif type == 'npy':
            xs, ys = parse_npy_with_leading_zero(os.path.join(dir, f), leading_zero)
        else:
            pdb.set_trace()

        if repeat_x is not None:
            assert repeat_x > xs[-1]
            if start_x is not None:
                last_id = np.where((np.array(xs)-start_x)>0)[0][0]
                loop_num = xs[-1] - xs[last_id]
                loop_time = (repeat_x - xs[-1]) // loop_num + 1
                cont_xs = np.concatenate([np.array(xs[last_id:])-xs[last_id-1]+xs[-1]+loop_num*i for i in range(int(loop_time))]).tolist()
                xs = xs + cont_xs
                ys = ys + ys[last_id:] * int(loop_time)
            else:
                require_num = repeat_x - xs[-1]
                last_id = np.where((np.array(xs)-xs[-1]+require_num)>0)[0][0]
                xs = xs + (np.array(xs[last_id:])-xs[last_id-1]+xs[-1]).tolist()
                ys = ys + ys[last_id:]

            xs.append(repeat_x)
            ys.append(np.max(ys))

        if y_fun is None:
            ys = [ratio * y for y in ys]
        else:
            ys = [y_fun(ratio*y) for y in ys]

        if "last" in f:
            xs, ys = smooth(xs, ys, k=5)
        else:
            xs, ys = smooth(xs, ys)

        
        list_xs.append(xs)
        if divide:
            list_ys.append([y / 100 for y in ys])
        else:
            list_ys.append(ys)
    return generate_plot_data(list_xs, list_ys)

def generate_wandb_data(dir, add_zero_x=None):
    wandb_files = os.listdir(dir)
    list_xs = []
    list_ys = []
    for f in wandb_files:
        xs, ys = parse_wandb_csv(os.path.join(dir, f))
        if add_zero_x is not None and xs[-1] < add_zero_x:
            xs.append(add_zero_x)
            ys.append(0)
        list_xs.append(xs)
        list_ys.append(ys)
    return generate_plot_data(list_xs, list_ys)

def generate_goalgail_data(dir):
    wandb_files = os.listdir(dir)
    list_xs = []
    list_ys = []
    for f in wandb_files:
        xs, ys = parse_goalgail_csv(os.path.join(dir, f))
        list_xs.append(xs)
        list_ys.append(ys)
    return generate_plot_data(list_xs, list_ys)

# def merge_curve(xs1, ys1, xs2, ys2, existing_curves):
#     # xs1, xs2 needs to be sorted
#     if len(xs1) == 0:
#         assert existing_curves == 0
#         return xs2, ys2, existing_curves + 1
    
#     for new_x, new_y in zip(xs2, ys2):
#         added = False
#         # first, find equal point
#         for i in range(0, len(xs1)):
#             if new_x == xs1[i]:
#                 ys1[i] = (ys1[i] * existing_curves + new_y) / (existing_curves + 1)
#                 added = True
#                 break

#         if new_x < xs1[0]:
#             add = True
#             pdb.set_trace()

#         if new_x > xs1[-1]:
#             add = True
#             xs1.append(new_x)
#             ys1.append(new_y * (existing_curves + 1))
        

#         if not added:
#             for i in range(0, len(xs1) - 1):
#                 if xs1[i] < new_x < xs1[i + 1]:
#                     added = True
#                     interp_value = np.interp(new_x, [xs1[i], xs1[i + 1]], [ys1[i], ys1[i+ 1]])
#                     interp_new_y = (interp_value * existing_curves + new_y) / (existing_curves + 1)
#                     xs1.insert(i + 1, new_x)
#                     ys1.insert(i + 1, interp_new_y)
#                     break

#         if not added:
#             pdb.set_trace()
    
#     return xs1, ys1, existing_curves + 1

def plot_with_std(ax, xs, ys, ystd, label, color, have_label):
    if have_label:
        ax.plot(xs, ys, label=label, color=color, alpha=0.7)
        ax.fill_between(xs, ys-ystd, ys+ystd, color=color, alpha=0.3)
    else:
        ax.plot(xs, ys, color=color, alpha=0.7)
        ax.fill_between(xs, ys-ystd, ys+ystd, color=color, alpha=0.3)

def plot_bc(ax, bc_value, label, color, have_label):
    if have_label:
        ax.plot([0, 4e7], [bc_value, bc_value], label=label, color=color, linestyle="dashed", alpha=0.7)
    else:
        ax.plot([0, 4e7], [bc_value, bc_value], color=color, linestyle="dashed", alpha=0.7)

to_color = {
    "bc": "blue",
    "gail": "red",
    "goalgail": "green",
    "deepset": "brown",
    "roboscribe": "purple"
}

def plot_data(ax, data, title, label=True, set_y_label=False, set_xlim=False, fig=None):
    if "bc" in data:
        plot_bc(ax, data["bc"], label="BC", color=to_color["bc"], have_label=label)

    if "gail" in data:
        xs, ys, ystd = data["gail"]
        plot_with_std(ax, xs, ys, ystd, label="GAIL", color=to_color["gail"], have_label=label)

    if "goalgail" in data:
        xs, ys, ystd = data["goalgail"]
        plot_with_std(ax, xs, ys, ystd, label="goalGAIL", color=to_color["goalgail"], have_label=label)

    if "deepset" in data:
        xs, ys, ystd = data["deepset"]
        plot_with_std(ax, xs, ys, ystd, label="DeepSet", color=to_color["deepset"], have_label=label)
    
    if "roboscribe" in data:
        xs, ys, ystd = data["roboscribe"]
        if fig is not None:
            do_plot(fig, ax, to_color["roboscribe"])
        else:
            plot_with_std(ax, xs, ys, ystd, label="RoboScribe", color=to_color["roboscribe"], have_label=label)

    ax.set_title(title, fontsize='xx-large', fontweight='bold')
    # ax.set_xlabel("environment steps", fontsize='x-large')
    ax.set_xlabel("Env. Steps", fontsize='x-large')
    if set_y_label:
        ax.set_ylabel("Success Rate", fontsize='x-large')
    ax.get_xaxis().get_offset_text().set_position((1.05,1.1))
    # ax.xaxis.set_label_coords(0.4, -0.077)
    # ax.set_xlim([0, min(data["roboscribe"][0][-1], 4e6)])

    

def get_tower1_data():
    dirname = "tower1"
    bc_value = 0.901
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_goalgail_data(os.path.join(dirname, "goalgail"))
    gail_xs, gail_ys, gail_ystd = generate_goalgail_data(os.path.join(dirname, "gail"))
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join(dirname, "deepset"))
    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd)}


def get_pnp1_data():
    dirname = "pnp1"
    bc_value = 0.848
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_goalgail_data(os.path.join(dirname, "goalgail"))
    gail_xs, gail_ys, gail_ystd = generate_goalgail_data(os.path.join(dirname, "gail"))
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join(dirname, "deepset"))
    robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe"), leading_zero=1.8e6, divide=False, repeat_x=4e6, start_x=2.3e6)
    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

def get_push1_data():
    dirname = "push1"
    bc_value = 0.79
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_wandb_data(os.path.join(dirname, "goalgail"))
    gail_xs, gail_ys, gail_ystd = generate_wandb_data(os.path.join(dirname, "gail"))
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join(dirname, "deepset"))
    robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe"), leading_zero=2.96e6, divide=False)
    
    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

def get_tower4_data():
    dirname = "tower4"
    bc_value = 0.0178
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_wandb_data(os.path.join(dirname, "goalgail"), add_zero_x=12879999)
    gail_xs, gail_ys, gail_ystd = generate_wandb_data(os.path.join(dirname, "gail"), add_zero_x=12879999)
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join(dirname, "deepset"), add_zero_x=12879999)
    # robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe"), leading_zero=3e6)
    robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe_npy"), leading_zero=3e6, type='npy')
    
    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

def get_tower5_data():
    dirname = "tower4"
    bc_value = 0.0178
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_wandb_data(os.path.join(dirname, "goalgail"), add_zero_x=16739998)
    gail_xs, gail_ys, gail_ystd = generate_wandb_data(os.path.join(dirname, "gail"), add_zero_x=16739998)
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join(dirname, "deepset"), add_zero_x=16739998)
    # robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe"), leading_zero=3e6)
    dirname = "tower5"
    robo_first_xs, robo_first_ys, robo_first_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe_npy_pre"), leading_zero=3e6, type='npy', ratio=14.0/86.0)

    cut_id = np.where(robo_first_xs >= 8e6)[0][0]
    # robo_first_xs, robo_first_ys, robo_first_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe_npy_pre"), leading_zero=3e6, type='npy')
    robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe_npy"), leading_zero=0, type='npy')

    robo_xs = np.concatenate([robo_first_xs[:cut_id], robo_xs+robo_first_xs[cut_id]], axis=0)
    robo_ys = np.concatenate([robo_first_ys[:cut_id], robo_ys], axis=0)
    robo_ystd = np.concatenate([robo_first_ystd[:cut_id], robo_ystd], axis=0)

    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

def get_pnp4_data():
    dirname = "pnp4"
    bc_value = 0.027
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_wandb_data(os.path.join(dirname, "goalgail"))
    gail_xs, gail_ys, gail_ystd = generate_wandb_data(os.path.join(dirname, "gail"))
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join(dirname, "deepset"))
    robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe"), leading_zero=2.4e6, repeat_x=4e6)

    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

def get_push3_data():
    dirname = "push3"
    bc_value = 0.35
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_wandb_data(os.path.join(dirname, "goalgail"), add_zero_x=8e6)
    gail_xs, gail_ys, gail_ystd = generate_wandb_data(os.path.join(dirname, "gail"), add_zero_x=8e6)
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join(dirname, "deepset_new_2"), add_zero_x=8e6)
    robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe_new"), leading_zero=2.96e6, divide=True, type='npy')
    # robo_ys = robo_ys * 3 / 9
    # robo_ystd = robo_ystd * (3/9) **2

    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

def get_manskill_data():
    dirname = "manskill"
    bc_value = 0.0
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_wandb_data(os.path.join("pnp4", "goalgail"), add_zero_x=18479999)
    gail_xs, gail_ys, gail_ystd = generate_wandb_data(os.path.join("pnp4", "gail"), add_zero_x=18479999)
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join("pnp4", "deepset"), add_zero_x=18479999)
    robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join(dirname, "roboscribe_npy"), leading_zero=4.4e6, divide=True, type='npy')
    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

def get_condition_data():
    dirname = "condition"
    bc_value = 0.489
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_wandb_data(os.path.join("pnp4", "goalgail"), add_zero_x=7e6)
    gail_xs, gail_ys, gail_ystd = generate_wandb_data(os.path.join("pnp4", "gail"), add_zero_x=7e6)
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join("pnp4", "deepset"), add_zero_x=7e6)
    robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join('pnp1', "roboscribe"), leading_zero=4.4e6, divide=False, y_fun=lambda x: x*x, repeat_x=7e6, start_x=4.7e6)
    # robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join('pnp1', "roboscribe"), leading_zero=4.4e6, divide=False, y_fun=lambda x: x*x)
    # pdb.set_trace()

    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

def get_meta_data():
    bc_value = 0.19/3
    goalgail_xs, goalgail_ys, goalgail_ystd = generate_wandb_data(os.path.join("pnp4", "goalgail"), add_zero_x=7e6)
    gail_xs, gail_ys, gail_ystd = generate_wandb_data(os.path.join("pnp4", "gail"), add_zero_x=7e6)
    dps_xs, dps_ys, dps_ystd = generate_wandb_data(os.path.join("pnp4", "deepset"), add_zero_x=7e6)
    # robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join('pnp1', "roboscribe"), leading_zero=4.4e6, divide=False, y_fun=lambda x: x*x, repeat_x=7e6, start_x=5.1e6)
    # robo_xs, robo_ys, robo_ystd = generate_leading_zero_data(os.path.join('pnp1', "roboscribe"), leading_zero=4.4e6, divide=False, y_fun=lambda x: x*x)
    robo_xs = robo_ys = robo_ystd = []
    # pdb.set_trace()

    return {"bc": bc_value, "goalgail": (goalgail_xs, goalgail_ys, goalgail_ystd), "gail": (gail_xs, gail_ys, gail_ystd), "deepset": (dps_xs, dps_ys, dps_ystd),
            "roboscribe": (robo_xs, robo_ys, robo_ystd)}

# pick&place-1, push-1, pick&place-4, push-3, tower-4, tower-5, pickcubedrawer, meta, if condition
def plot():
    fig, ax = plt.subplots(2, 4, figsize=(12, 7.6))
    # plot_data(ax[0], get_tower1_data(), "Tower-1", set_y_label=True)
    plot_data(ax[0][0], get_pnp1_data(), "Pick&Place-1", set_y_label=True)
    plot_data(ax[0][1], get_pnp4_data(), "Pick&Place-4", label=False)
    plot_data(ax[0][2], get_condition_data(), "Pick&Place-Cond", label=False)
    plot_data(ax[0][3], get_push3_data(), "Push-3", label=False)
    plot_data(ax[1][0], get_tower5_data(), "Tower-5", set_y_label=True, label=False)
    plot_data(ax[1][1], get_meta_data(), "Meta-World", label=False, fig=fig)
    plot_data(ax[1][2], get_manskill_data(), "PlaceCubesDrawer", label=False)

    # plot_data(ax[1][4], get_manskill_data(), "PickCubeDrawer", label=False)

    # plot_data(ax[3], get_push3_data(), "Push-3", label=False)
    ax[0][0].set_xlim([0, 4e6])
    ax[0][1].set_xlim([0, 4e6])
    ax[0][2].set_xlim([0, 6e6])
    ax[0][3].set_xlim([0, 8e6])
    ax[1][0].set_xlim([0, 16739998])
    ax[1][1].set_xlim([0, 6e6])
    ax[1][2].set_xlim([0, 18479999])

    ax[0][0].set_ylim([-0.04, 1.0])
    ax[0][1].set_ylim([-0.04, 1.0])
    ax[0][2].set_ylim([-0.04, 1.0])
    ax[0][3].set_ylim([-0.04, 1.0])
    ax[1][0].set_ylim([-0.04, 1.0])
    ax[1][1].set_ylim([-0.04, 1.0])
    ax[1][2].set_ylim([-0.04, 1.0])

    fig.delaxes(ax[1][3])
    fig.subplots_adjust(bottom=0.17)
    fig.legend(ncols=5, loc='lower center', frameon=False, fontsize="x-large")
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    # fig.tight_layout()
    fig.savefig("a.pdf")

def plot_manskill():
    fig = plt.figure()
    ax = fig.gca()
    xs, ys, ystd = generate_wandb_data("manskill")
    plot_with_std(ax, xs, ys, ystd, label="GAIL", color="purple", have_label=True)
    fig.savefig("b.png")
    
    


if __name__ == "__main__":
    # get_tower1_data()
    plot()
    # plot_manskill()