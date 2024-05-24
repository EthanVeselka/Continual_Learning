import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Standard deviation AUC-ROC Macro
base_avg = np.array(
    [
        [
            [0.83617324, 0.46029995],
            [0.76235228, 0.30406],
            [0.76734919, 0.26124697],
            [0.72895897, 0.24318802],
            [0.84965742, 0.50144148],
        ],
        [
            [0.80343268, 0.38051104],
            [0.88248171, 0.61737701],
            [0.84262209, 0.52656855],
            [0.84579313, 0.6219254],
            [0.89969604, 0.69365603],
        ],
        [
            [0.7946909, 0.36386452],
            [0.87891624, 0.60767766],
            [0.8711796, 0.56341899],
            [0.86125169, 0.63691693],
            [0.89887884, 0.70173367],
        ],
        [
            [0.78406671, 0.35693945],
            [0.87019445, 0.59729956],
            [0.8614086, 0.54904338],
            [0.88180151, 0.67013995],
            [0.89686823, 0.71641679],
        ],
        [
            [0.79128877, 0.35212925],
            [0.88960036, 0.6221493],
            [0.86568478, 0.55661669],
            [0.86990169, 0.65481245],
            [0.91321972, 0.73214236],
        ],
    ]
)

comb_avg = np.array(
    [
        [
            [0.83179128, 0.45880586],
            [0.76366943, 0.29675906],
            [0.7779752, 0.2743554],
            [0.73220563, 0.2651969],
            [0.86035204, 0.50938213],
        ],
        [
            [0.82467255, 0.44396378],
            [0.88162158, 0.61421214],
            [0.84135813, 0.52676893],
            [0.84580724, 0.60580093],
            [0.90704587, 0.70226288],
        ],
        [
            [0.82256443, 0.44159848],
            [0.89017498, 0.63464909],
            [0.87828178, 0.58400307],
            [0.86889669, 0.64595691],
            [0.90278048, 0.70859848],
        ],
        [
            [0.81764967, 0.43249619],
            [0.88611181, 0.62287828],
            [0.87238064, 0.57578643],
            [0.89201824, 0.68301501],
            [0.90970426, 0.72551745],
        ],
        [
            [0.82152584, 0.43115608],
            [0.89244733, 0.63509023],
            [0.8790842, 0.57125006],
            [0.8824038, 0.66155962],
            [0.91550191, 0.73557102],
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


make_tables(base_avg, "ihm_baseline", "AUC-ROC", "AUC-PR")
make_tables(comb_avg, "ihm_comb", "AUC-ROC", "AUC-PR")
make_tables(comb_avg - base_avg, "ihm_diff", "AUC-ROC", "AUC-PR")
print(f"Per Task Average: {[np.mean(base_avg[i][0:i+1,:], axis=0) for i in range(5)]}")
print(f"Per Task Average: {[np.mean(comb_avg[i][0:i+1,:], axis=0) for i in range(5)]}")
