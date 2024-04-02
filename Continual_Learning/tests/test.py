import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from test_standard_lstm import TestLSTM

# ssh veselka@cse-stmi-s1.cse.tamu.edu
# buffer size must be <= samplesize/train_batch_size = 1000/8 = 125

# python3 test.py --bl --test --rt --i 5 --n 5   # Run baseline tests, 5 iterations, and report best 5 models ranked by test results, as well as averages, std_dev, conf matrix
# python3 test.py --tasks 2 --ihm --grid --n 3   # Run gridsearch on ihm with 2 task, report best 3 models at each buffer size (X)
# python3 test.py --tasks 5 --ihm --grid --n 3   # Run gridsearch on ihm with 5 tasks, report best 3 models at each buffer size (rerun with corrections)
# python3 test.py --tasks 5 --phen --grid --n 3  # Run gridsearch on phen with 2 task, report best 3 models at each buffer size (get splits, run for splits)

# add metric avg perf of tasks up through n, not including past tasks
# organize results into a confusion matrix (test results, 5x5 grid, either one for each metric, or just put both)
# rank grid search models by best average primary metric as is already being done, test best of each buffer size 3 times
# graph buffer size vs perf of best models

random.seed(42)

param_grid = {
    "buffer_size": [500, 425, 350, 275, 200, 125, 50],
    "Importance": [8, 6, 4, 2],
    "ihm_epochs": 4,
    "phen_epochs": 6,
    "los_epochs": 1,
    "dec_epochs": 1,
}

ihm_perf = []
ihm_split_perf = []
phen_perf = []
los_perf = []
dec_perf = []


def get_best_perf(n, k, task_perf, name, num_tasks):
    m1_performances = []

    for _, perf in enumerate(task_perf):
        p = perf[k]
        metric1 = p[0][0]
        metric2 = p[0][1]

        v1 = p[1]["Final Average " + metric1]
        v2 = p[1]["Final Average " + metric2]
        conf = p[2]
        if k == "test":
            avgs = {}
            for num in range(num_tasks):
                avgs[f"Task {num+1}"] = {
                    "Average " + metric1: p[1][f"Task {num+1} Average " + metric1],
                    "Average " + metric2: p[1][f"Task {num+1} Average " + metric2],
                }
        m1_performances.append((metric1, v1, metric2, v2, conf, avgs))

    if k == "test":
        data = []
        for _, perf in enumerate(task_perf):
            p = perf[k]
            v3 = p[1]["Scores"]
            perf_data = []
            for task_num, task in enumerate(v3):
                t = task["{Task " + str(task_num + 1) + "}"]
                vals = []
                for i, eval_task in enumerate(t):
                    m1 = eval_task["Eval Task " + str(i + 1)][metric1]
                    m2 = eval_task["Eval Task " + str(i + 1)][metric2]
                    vals.append([m1, m2])

                perf_data.append(vals)

            data.append(np.array(perf_data))

        averages = np.mean(data, axis=0)
        stacked = np.stack(data, axis=-1)
        std_dev = np.std(stacked, axis=-1)
        std_dev_m1 = std_dev[:, :, 0]
        std_dev_m2 = std_dev[:, :, 1]
        m = averages.shape[0]

        first_values = averages[:, :, 0]
        second_values = averages[:, :, 1]

        tasks = [f"Task {i}" for i in range(1, m + 1)]
        df1 = pd.DataFrame(first_values, tasks, tasks)
        df2 = pd.DataFrame(second_values, tasks, tasks)

        # Plot confusion matrices
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        sns.heatmap(df1, annot=True, cmap="Blues", fmt=".3f")
        plt.yticks(rotation=0)
        plt.xlabel("Evaluated")
        plt.ylabel("Trained")
        plt.gca().xaxis.set_ticks_position("top")
        plt.gca().xaxis.set_label_position("top")
        plt.title("Trained vs. Evaluated " + metric1, pad=20)

        plt.subplot(1, 2, 2)
        sns.heatmap(df2, annot=True, cmap="Blues", fmt=".3f")
        plt.yticks(rotation=0)
        plt.xlabel("Evaluated")
        plt.ylabel("Trained")
        plt.gca().xaxis.set_ticks_position("top")
        plt.gca().xaxis.set_label_position("top")
        plt.title("Trained vs. Evaluated " + metric2, pad=20)

        plt.savefig("../results/" + name + ".png")

    m1sorted = sorted(m1_performances, key=lambda x: x[1], reverse=True)
    if n > len(m1sorted):
        n = len(m1sorted)

    with open("../results/" + name + ".txt", "w") as f:
        print(f"Best {k} performances:", file=f)
        print("----------------------------------", file=f)
        for idx, model in enumerate(m1sorted[:n]):
            print(
                f"Model: Final Average {model[0]}: {model[1]}, Final Average {model[2]}: {model[3]}\nConfiguration: {model[4]}",
                file=f,
            )
            print("\n", file=f)

        if k == "test":
            print(f"Per Task Average: {m1sorted[0][5]}", file=f)
            print("\n", file=f)
            print("Average performance:\n", averages, file=f)
            print("\n", file=f)
            print("Standard deviation " + metric1 + ":\n", std_dev_m1, file=f)
            print("\n", file=f)
            print("Standard deviation " + metric2 + ":\n", std_dev_m2, file=f)


def gridsearch(n, k, task, task_list):
    print("---------------------------------------")
    print("Initiating grid search for " + task + "...")
    print("---------------------------------------")
    print("\n")
    if task == "ihm":
        epochs = param_grid["ihm_epochs"]
    elif task == "phen":
        epochs = param_grid["phen_epochs"]
    elif task == "los":
        epochs = param_grid["los_epochs"]
    elif task == "decomp":
        epochs = param_grid["dec_epochs"]

    for bs in param_grid["buffer_size"]:
        task_perf = []

        print("\n")
        print("Replay")
        print(f"Buffer size: {bs}")
        print("---------------------------------------")
        task_perf.append(
            TestLSTM().test_standard_lstm(
                task,
                epochs,
                task_list,
                buffer_size=bs,
                replay=True,
                ewc_penalty=False,
                importance=0,
            )
        )
        for imp in param_grid["Importance"]:
            print("\n")
            print("EWC")
            print(f"Buffer size: {bs}, Importance: {imp}")
            print("---------------------------------------")
            task_perf.append(
                TestLSTM().test_standard_lstm(
                    task,
                    epochs,
                    task_list,
                    buffer_size=bs,
                    replay=False,
                    ewc_penalty=True,
                    importance=imp,
                )
            )
            print("\n")
            print("Both")
            print(f"Buffer size: {bs}, Importance: {imp}")
            print("---------------------------------------")
            task_perf.append(
                TestLSTM().test_standard_lstm(
                    task,
                    epochs,
                    task_list,
                    buffer_size=bs,
                    replay=True,
                    ewc_penalty=True,
                    importance=imp,
                )
            )

        get_best_perf(
            n, k, task_perf, f"{task}_{str(bs)}_{str(len(task_list))}", len(task_list)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", type=int, help="Run tasks 1 through <task#>, defaults to 1"
    )
    parser.add_argument(
        "--b",
        type=int,
        help="Specifies buffer size (in number of batches) to be used in EWC and Replay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--imp",
        type=int,
        help="Specifies EWC importance",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Use replay, must specify > 1 task using --tasks",
    )
    parser.add_argument(
        "--ewc",
        action="store_true",
        help="Use EWC penalty, must specify > 1 task using --tasks",
    )
    parser.add_argument(
        "--ihm",
        action="store_true",
        help="Test for IHM task",
    )
    parser.add_argument(
        "--dec",
        action="store_true",
        help="Test for Decompensation task",
    )
    parser.add_argument(
        "--los",
        action="store_true",
        help="Test for LoS task",
    )
    parser.add_argument(
        "--phen",
        action="store_true",
        help="Test for Phenotyping task",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run grid search for hyperparameters",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test final model",
    )
    parser.add_argument(
        "--bl",
        action="store_true",
        help="Run baseline tests",
    )
    parser.add_argument(
        "--rt",
        action="store_true",
        help="Rank results by test or validation performance",
    )
    parser.add_argument(
        "--i",
        type=int,
        help="Number of iterations to run",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Save top n model performances",
    )

    args = parser.parse_args()

    task_list = [i for i in range(0, args.tasks)] if args.tasks else [0]
    num_tasks = len(task_list)
    iterations = args.i if args.i else 1
    buffer_size = args.b if (args.ewc or args.replay) else 0
    importance = args.imp if args.ewc else 0
    replay = args.replay if (args.replay and buffer_size > 0) else False
    ewc_penalty = args.ewc if (args.ewc and buffer_size > 0) else False
    k = "test" if args.rt else "val"
    n = args.n if args.n else 5

    if args.test:
        test = True
    else:
        test = False

    ihm_perf = []
    ihm_split_perf = []
    phen_perf = []
    los_perf = []
    dec_perf = []

    if args.ihm:
        epochs = args.epochs if args.epochs else param_grid["ihm_epochs"]
        if args.grid:
            gridsearch(n, "val", "ihm", task_list)
            return
        print("---------------------------------------")
        print("Testing standard LSTM on IHM dataset...")
        print("---------------------------------------")
        print("\n")
        for i in range(iterations):
            ihm_perf.append(
                TestLSTM().test_standard_lstm(
                    "ihm",
                    epochs,
                    task_list,
                    buffer_size,
                    replay,
                    ewc_penalty,
                    importance,
                    test=test,
                )
            )

        get_best_perf(n, k, ihm_perf, "ihm_perf", num_tasks)
        return

    if args.phen:
        epochs = args.epochs if args.epochs else param_grid["phen_epochs"]
        if args.grid:
            gridsearch(n, "val", "phen", task_list)
            return
        print("-----------------------------------------------")
        print("Testing standard LSTM on Phenotyping dataset...")
        print("-----------------------------------------------")
        print("\n")
        for i in range(iterations):
            phen_perf.append(
                TestLSTM().test_standard_lstm(
                    "phen",
                    epochs,
                    task_list,
                    buffer_size,
                    replay,
                    ewc_penalty,
                    importance,
                    test=test,
                )
            )

        get_best_perf(n, k, phen_perf, "phen_perf", num_tasks)
        return

    if args.los:
        epochs = args.epochs if args.epochs else param_grid["los_epochs"]
        if args.grid:
            gridsearch(n, "val", "los", task_list)
            return
        print("---------------------------------------")
        print("Testing standard LSTM on LoS dataset...")
        print("---------------------------------------")
        print("\n")
        for i in range(iterations):
            los_perf.append(
                TestLSTM().test_standard_lstm(
                    "los",
                    epochs,
                    task_list,
                    buffer_size,
                    replay,
                    ewc_penalty,
                    importance,
                    test=test,
                )
            )

        get_best_perf(n, k, los_perf, "los_perf", num_tasks)
        return

    if args.dec:
        epochs = args.epochs if args.epochs else param_grid["dec_epochs"]
        if args.grid:
            gridsearch(n, "val", "decomp", task_list)
            return
        print("--------------------------------------------------")
        print("Testing standard LSTM on Decompensation dataset...")
        print("--------------------------------------------------")
        print("\n")
        for i in range(iterations):
            dec_perf.append(
                TestLSTM().test_standard_lstm(
                    "decomp",
                    epochs,
                    task_list,
                    buffer_size,
                    replay,
                    ewc_penalty,
                    importance,
                    test=test,
                )
            )

        get_best_perf(n, k, dec_perf, "dec_perf", num_tasks)
        return

    if args.bl:
        print("--------------------------------------")
        print("Testing standard LSTM for baselines...")
        print("--------------------------------------")
        print("\n")
        for i in range(iterations):
            ihm_perf.append(
                TestLSTM().test_standard_lstm(
                    "ihm",
                    param_grid["ihm_epochs"],
                    task_list,
                    buffer_size=0,
                    replay=False,
                    ewc_penalty=False,
                    importance=False,
                    test=True,
                )
            )
            phen_perf.append(
                TestLSTM().test_standard_lstm(
                    "phen",
                    param_grid["phen_epochs"],
                    task_list,
                    buffer_size=0,
                    replay=False,
                    ewc_penalty=False,
                    importance=False,
                    test=True,
                )
            )
        get_best_perf(n, k, ihm_perf, "ihm_baseline", num_tasks)
        get_best_perf(n, k, phen_perf, "phen_baseline", num_tasks)
        return


main()
