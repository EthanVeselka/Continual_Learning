import argparse
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from test_standard_lstm import TestLSTM

# grid search should be good for IHM
# need to implement/test buffer updating method
# (keep static buffer, initially filled with buff_size random samples, rebalance after each task)
# might adjust data loading method to be sequential (load data for current task, use to update buffer)
# load testing data for all tasks at once, neccessary for evaluating EWC and Replay, but training data can be done sequentially
# write testing script (real testing)

# ssh veselka@cse-stmi-s1.cse.tamu.edu
# buffer size must be <= samplesize/train_batch_size = 1000/8 = 125

param_grid = {
    "buffer_size": [125, 200, 275, 350, 425, 500, 750, 1000],
    "Importance": [1, 2, 4, 6, 8],
}


def ihmgridsearch(task_list):
    print("---------------------------------------")
    print("Initiating grid search for IHM...")
    print("---------------------------------------")
    print("\n")
    for bs in param_grid["buffer_size"]:
        print("\n")
        print("Replay")
        print(f"Buffer size: {bs}, Importance: {imp}")
        print("---------------------------------------")
        TestLSTM().test_standard_lstm_ihm(task_list, bs, True, False, 0)
        for imp in param_grid["Importance"]:
            print("\n")
            print("EWC")
            print(f"Buffer size: {bs}, Importance: {imp}")
            print("---------------------------------------")
            TestLSTM().test_standard_lstm_ihm(task_list, bs, False, True, imp)
            print("\n")
            print("Both")
            print(f"Buffer size: {bs}, Importance: {imp}")
            print("---------------------------------------")
            TestLSTM().test_standard_lstm_ihm(task_list, bs, True, True, imp)


def phengridsearch(task_list):
    print("-----------------------------------------")
    print("Initiating grid search for Phenotyping...")
    print("-----------------------------------------")
    print("\n")
    for bs in param_grid["buffer_size"]:
        print("\n")
        print("Replay")
        print(f"Buffer size: {bs}, Importance: {imp}")
        print("---------------------------------------")
        TestLSTM().test_standard_lstm_phenotype(task_list, bs, True, False, 0)
        for imp in param_grid["Importance"]:
            print("\n")
            print("EWC")
            print(f"Buffer size: {bs}, Importance: {imp}")
            print("---------------------------------------")
            TestLSTM().test_standard_lstm_phenotype(task_list, bs, False, True, imp)
            print("\n")
            print("Both")
            print(f"Buffer size: {bs}, Importance: {imp}")
            print("---------------------------------------")
            TestLSTM().test_standard_lstm_phenotype(task_list, bs, True, True, imp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", type=int, required=True, help="Run tasks 1 through <task#>"
    )
    parser.add_argument(
        "--b",
        type=int,
        help="Specifies buffer size (in number of batches) to be used in EWC and Replay",
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
        "--all",
        action="store_true",
        help="Test for all tasks",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run grid search for hyperparameters",
    )

    args = parser.parse_args()

    task_list = [i for i in range(0, args.tasks)]
    buffer_size = args.b if (args.ewc or args.replay) else 0
    importance = args.imp if args.ewc else 0
    replay = args.replay if (args.replay and buffer_size > 0) else False
    ewc_penalty = args.ewc if (args.ewc and buffer_size > 0) else False

    if args.ihm:
        if args.grid:
            ihmgridsearch(task_list)
            return
        print("---------------------------------------")
        print("Testing standard LSTM on IHM dataset...")
        print("---------------------------------------")
        print("\n")
        TestLSTM().test_standard_lstm_ihm(
            task_list, buffer_size, replay, ewc_penalty, importance
        )

        return

    if args.dec:
        if args.grid:

            return
        print("--------------------------------------------------")
        print("Testing standard LSTM on Decompensation dataset...")
        print("--------------------------------------------------")
        print("\n")
        TestLSTM().test_standard_lstm_decomp(
            task_list, buffer_size, replay, ewc_penalty
        )

        return

    if args.los:
        if args.grid:

            return
        print("---------------------------------------")
        print("Testing standard LSTM on LoS dataset...")
        print("---------------------------------------")
        print("\n")
        TestLSTM().test_standard_lstm_los()

    if args.phen:
        if args.grid:
            phengridsearch(task_list)
            return
        print("-----------------------------------------------")
        print("Testing standard LSTM on Phenotyping dataset...")
        print("-----------------------------------------------")
        print("\n")
        TestLSTM().test_standard_lstm_phenotype(
            task_list, buffer_size, replay, ewc_penalty, importance
        )

    if args.all:
        print("---------------------------------------")
        print("Testing standard LSTM on all datasets...")
        print("---------------------------------------")
        print("\n")
        TestLSTM().test_standard_lstm_ihm(task_list, buffer_size, replay, ewc_penalty)
        TestLSTM().test_standard_lstm_decomp(
            task_list, buffer_size, replay, ewc_penalty
        )
        TestLSTM().test_standard_lstm_los(task_list, buffer_size, replay, ewc_penalty)
        TestLSTM().test_standard_lstm_phenotype(
            task_list, buffer_size, replay, ewc_penalty
        )

        return


main()
