import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


base_avg = np.array(
    [
        [
            [0.8933093, 0.26894212],
            [0.65021018, 0.04985296],
            [0.64806734, 0.04132016],
            [0.63696256, 0.07485634],
            [0.75064714, 0.07517501],
        ],
        [
            [0.84857422, 0.21980533],
            [0.78146646, 0.1405915],
            [0.70133892, 0.0793111],
            [0.69730633, 0.12804746],
            [0.80479231, 0.14161287],
        ],
        [
            [0.82838147, 0.19471608],
            [0.76093049, 0.11919184],
            [0.70902953, 0.07576243],
            [0.69521851, 0.11711063],
            [0.79919591, 0.15067971],
        ],
        [
            [0.80609796, 0.18017887],
            [0.7860822, 0.12978396],
            [0.71969477, 0.07781627],
            [0.72765574, 0.12718273],
            [0.81196035, 0.14874395],
        ],
        [
            [0.85811414, 0.19715228],
            [0.80876611, 0.1496446],
            [0.72597068, 0.07891485],
            [0.72907851, 0.14399428],
            [0.84071422, 0.1657538],
        ],
    ]
)

ewc_avg = np.array(
    [
        [
            [0.89079031, 0.27186735],
            [0.66254297, 0.0480195],
            [0.65730409, 0.0457468],
            [0.64617883, 0.08277716],
            [0.75463671, 0.06925873],
        ],
        [
            [0.87232314, 0.25407199],
            [0.79560103, 0.15612159],
            [0.69459269, 0.08113758],
            [0.69800052, 0.12690021],
            [0.79555652, 0.14054222],
        ],
        [
            [0.84642322, 0.23122493],
            [0.77127897, 0.13229341],
            [0.70431002, 0.08210294],
            [0.68974294, 0.12233162],
            [0.79860412, 0.15381368],
        ],
        [
            [0.85086466, 0.22473718],
            [0.78992497, 0.13474093],
            [0.72477396, 0.09185725],
            [0.72697394, 0.13489423],
            [0.81424095, 0.14854024],
        ],
        [
            [0.87435264, 0.21948894],
            [0.80974965, 0.15346681],
            [0.72611501, 0.086776],
            [0.72780077, 0.14387259],
            [0.84660894, 0.17290318],
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
    # plt.figure(figsize=(22, 7.75))
    sns.set(font_scale=1.5)
    plt.figure(figsize=(13, 10))
    # plt.subplot(1, 2, 1)
    # plt.figure()
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


make_tables(base_avg, "dec_baseline", "AUC-ROC", "AUC-PR")
make_tables(ewc_avg, "dec_EWC", "AUC-ROC", "AUC-PR")
make_tables(ewc_avg - base_avg, "dec_diff", "AUC-ROC", "AUC-PR")
print(f"Per Task Average: {[np.mean(base_avg[i][0:i+1,:], axis=0) for i in range(5)]}")
print(f"Per Task Average: {[np.mean(ewc_avg[i][0:i+1,:], axis=0) for i in range(5)]}")
