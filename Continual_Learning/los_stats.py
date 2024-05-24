import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_split_stats(task):
    df = pd.read_csv(
        f"../datasets/mimic3-benchmarks/length-of-stay/{task}_listfile.csv"
    )
    # df2 = pd.read_csv(f"length-of-stay_split/northeast_{task}.csv")
    # df3 = pd.read_csv(f"length-of-stay_split/south_{task}.csv")
    # df4 = pd.read_csv(f"length-of-stay_split/west_{task}.csv")

    # df = pd.concat([df1, df2, df3, df4], axis=0)
    df.reset_index(drop=True, inplace=True)
    # Extract the last column containing hours
    hours_column = df.iloc[:, -1]
    hours_column = hours_column[hours_column <= 200]
    maxhrs = int(hours_column.max() + 1)

    print(maxhrs)
    # Plot histogram
    plt.hist(
        hours_column, bins=int(maxhrs / 15), color="skyblue"
    )  # Assuming hours are from 0 to 23
    plt.xlabel("Hour")
    plt.ylabel("Frequency")
    plt.title("LoS y_true Distribution")
    # plt.xticks(range(maxhrs))  # Assuming hours are from 0 to 23
    plt.grid(axis="y", alpha=0.75)
    plt.savefig(f"{task}_histogram.png")


show_split_stats("train")
show_split_stats("test")
show_split_stats("val")
