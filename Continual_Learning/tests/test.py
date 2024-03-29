import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from test_standard_lstm import TestLSTM

# ssh veselka@cse-stmi-s1.cse.tamu.edu
# buffer size must be <= samplesize/train_batch_size = 1000/8 = 125

param_grid = {
    "buffer_size": [0, 1000, 750, 500, 425, 350, 275, 200, 125],
    "Importance": [8, 6, 4, 2, 1],
    "epochs": [8],
}


def gridsearch(task, task_list):
    print("---------------------------------------")
    print("Initiating grid search for IHM...")
    print("---------------------------------------")
    print("\n")
    for epochs in param_grid["epochs"]:
        for bs in param_grid["buffer_size"]:
            print("\n")
            print("Replay")
            print(f"Buffer size: {bs}")
            print("---------------------------------------")
            TestLSTM().test_standard_lstm(
                task,
                epochs,
                task_list,
                buffer_size=bs,
                replay=True,
                ewc_penalty=False,
                importance=0,
            )
            for imp in param_grid["Importance"]:
                print("\n")
                print("EWC")
                print(f"Buffer size: {bs}, Importance: {imp}")
                print("---------------------------------------")
                TestLSTM().test_standard_lstm(
                    task,
                    epochs,
                    task_list,
                    buffer_size=bs,
                    replay=False,
                    ewc_penalty=True,
                    importance=imp,
                )
                print("\n")
                print("Both")
                print(f"Buffer size: {bs}, Importance: {imp}")
                print("---------------------------------------")
                TestLSTM().test_standard_lstm(
                    task,
                    epochs,
                    task_list,
                    buffer_size=bs,
                    replay=True,
                    ewc_penalty=True,
                    importance=imp,
                )


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
        "--all",
        action="store_true",
        help="Test for all tasks",
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

    args = parser.parse_args()

    task_list = [i for i in range(0, args.tasks)]
    epochs = args.epochs if args.epochs else 4
    buffer_size = args.b if (args.ewc or args.replay) else 0
    importance = args.imp if args.ewc else 0
    replay = args.replay if (args.replay and buffer_size > 0) else False
    ewc_penalty = args.ewc if (args.ewc and buffer_size > 0) else False
    if args.test:
        test = True
    else:
        test = False

    if args.ihm:
        if args.grid:
            gridsearch("ihm", task_list)
            return
        print("---------------------------------------")
        print("Testing standard LSTM on IHM dataset...")
        print("---------------------------------------")
        print("\n")
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

        return

    if args.phen:
        if args.grid:
            gridsearch("phen", task_list)
            return
        print("-----------------------------------------------")
        print("Testing standard LSTM on Phenotyping dataset...")
        print("-----------------------------------------------")
        print("\n")
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

    if args.los:
        if args.grid:
            gridsearch("los", task_list)
            return
        print("---------------------------------------")
        print("Testing standard LSTM on LoS dataset...")
        print("---------------------------------------")
        print("\n")
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

    if args.dec:
        if args.grid:
            gridsearch("decomp", task_list)
            return
        print("--------------------------------------------------")
        print("Testing standard LSTM on Decompensation dataset...")
        print("--------------------------------------------------")
        print("\n")
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

        return

    if args.all:
        print("---------------------------------------")
        print("Testing standard LSTM on all datasets...")
        print("---------------------------------------")
        print("\n")
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

        return


main()
