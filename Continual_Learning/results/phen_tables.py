import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Standard deviation AUC-ROC Macro

# Average performance
base_avg = np.array(
    [
        [
            [0.76389079, 0.81487482],
            [0.46942806, 0.50788906],
            [0.47505469, 0.5183919],
            [0.48111941, 0.49985916],
            [0.4773876, 0.55115418],
        ],
        [
            [0.57470287, 0.60575977],
            [0.72001658, 0.80672446],
            [0.67177943, 0.77412103],
            [0.65471812, 0.75889595],
            [0.66478995, 0.76413686],
        ],
        [
            [0.55711812, 0.58587273],
            [0.68217196, 0.76506134],
            [0.71474375, 0.81589383],
            [0.66195582, 0.75590202],
            [0.62400334, 0.70876336],
        ],
        [
            [0.54214642, 0.58450601],
            [0.66473439, 0.75925551],
            [0.67892845, 0.77837739],
            [0.72198563, 0.81406616],
            [0.63477088, 0.73501261],
        ],
        [
            [0.54811638, 0.57898726],
            [0.67725876, 0.76141993],
            [0.65870381, 0.76400445],
            [0.63422251, 0.73139001],
            [0.71575981, 0.81731508],
        ],
    ]
)

# Standard deviation AUC-ROC Macro
comb_avg = np.array(
    [
        [
            [0.76394649, 0.81533948],
            [0.4870254, 0.5233567],
            [0.48328968, 0.52996043],
            [0.48780552, 0.50243994],
            [0.48237604, 0.56024777],
        ],
        [
            [0.74453748, 0.80024578],
            [0.70955845, 0.79943405],
            [0.65914129, 0.76359116],
            [0.63965403, 0.75196122],
            [0.65474059, 0.7561228],
        ],
        [
            [0.72528447, 0.78334715],
            [0.69603683, 0.77773508],
            [0.7138712, 0.8141036],
            [0.65986107, 0.76158935],
            [0.65718697, 0.74080254],
        ],
        [
            [0.72216409, 0.7808455],
            [0.67217217, 0.76615287],
            [0.68254848, 0.78332163],
            [0.72021416, 0.81292829],
            [0.64362972, 0.7375971],
        ],
        [
            [0.71830843, 0.77788441],
            [0.69714412, 0.7775764],
            [0.68508014, 0.78384769],
            [0.67020715, 0.77698824],
            [0.71988241, 0.81557007],
        ],
    ]
)


def make_tables(averages, name, metric1, metric2):
    n = ""
    c = None
    cmap = "Blues"
    if name.split("_")[1] == "baseline":
        n = "Baseline: "
    elif name.split("_")[1] == "diff":
        c = 0
        # cmap = sns.diverging_palette(10, 245, n=2, as_cmap=True)
        cmap = "vlag_r"
        n = "Differences: "
    elif name.split("_")[1] == "comb":
        n = "EWC & Replay: "
    elif name.split("_")[1] == "EWC":
        n = "EWC only: "
    elif name.split("_")[1] == "rep":
        n = "Replay only: "

    m = averages.shape[0]
    first_values = averages[:, :, 0].round(3)
    second_values = averages[:, :, 1].round(3)
    first_values[first_values == -0.0] = 0.0
    second_values[second_values == -0.0] = 0.0

    tasksx = [f"Task {i}" for i in range(1, m + 1)]
    tasksy = ["Task 1"]
    tasksy += [f"+ Task {i}" for i in range(2, m + 1)]
    df1 = pd.DataFrame(first_values, tasksy, tasksx)
    df2 = pd.DataFrame(second_values, tasksy, tasksx)

    # Plot confusion matrices
    sns.set(font_scale=1.5)
    plt.figure(figsize=(13, 10))

    sns.heatmap(df1, annot=True, cmap=cmap, center=c, fmt=".3f", annot_kws={"size": 18})
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel("Evaluated", labelpad=10, fontsize=18)
    plt.ylabel("Trained", fontsize=18)
    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().xaxis.set_label_position("top")
    plt.xticks(fontsize=18)
    plt.title(n + metric1, pad=15, fontsize=22)
    plt.savefig(name + "_" + metric1 + ".png", dpi=500)

    plt.figure(figsize=(13, 10))
    sns.heatmap(df2, annot=True, cmap=cmap, center=c, fmt=".3f", annot_kws={"size": 18})
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel("Evaluated", labelpad=10, fontsize=18)
    plt.ylabel("Trained", fontsize=18)
    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().xaxis.set_label_position("top")
    plt.xticks(fontsize=18)
    plt.title(n + metric2, pad=15, fontsize=22)
    plt.savefig(name + "_" + metric2 + ".png", dpi=500)


make_tables(base_avg, "phen_baseline", "AUC-ROC Macro", "AUC-ROC Micro")
make_tables(comb_avg, "phen_comb", "AUC-ROC Macro", "AUC-ROC Micro")
make_tables(comb_avg - base_avg, "phen_diff", "AUC-ROC Macro", "AUC-ROC Micro")
print(f"Per Task Average: {[np.mean(base_avg[i][0:i+1,:], axis=0) for i in range(5)]}")
print(f"Per Task Average: {[np.mean(comb_avg[i][0:i+1,:], axis=0) for i in range(5)]}")
