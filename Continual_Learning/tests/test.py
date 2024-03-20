import argparse
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from test_standard_lstm import TestLSTM

# rsync -av --ignore-existing ./datasets/eICU-benchmarks veselka@cse-stmi-s1.cse.tamu.edu:~/datasets/   # continue copying data, stopped in output-copy/2908432...
# ssh veselka@cse-stmi-s1.cse.tamu.edu
# buffer size must be <= samplesize/train_batch_size = 1000/8 = 125


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", type=int, required=True, help="Run tasks 1 through <task#>"
    )
    parser.add_argument(
        "--b",
        type=int,
        required=True,
        help="Specifies buffer size (in number of batches) to be used in EWC and Replay",
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
        "--all",
        action="store_true",
        help="Test for all tasks",
    )

    args = parser.parse_args()

    task_list = [i for i in range(0, args.tasks)]
    buffer_size = args.b
    replay = args.replay if (args.replay and buffer_size > 0) else False
    ewc_penalty = args.ewc if (args.ewc and buffer_size > 0) else False

    # torch.cuda.empty_cache()
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

    if args.ihm:
        print("---------------------------------------")
        print("Testing standard LSTM on IHM dataset...")
        print("---------------------------------------")
        print("\n")
        TestLSTM().test_standard_lstm_ihm(task_list, buffer_size, replay, ewc_penalty)

        return

    if args.dec:
        print("--------------------------------------------------")
        print("Testing standard LSTM on Decompensation dataset...")
        print("--------------------------------------------------")
        print("\n")
        TestLSTM().test_standard_lstm_decomp(
            task_list, buffer_size, replay, ewc_penalty
        )

        return

    if args.los:
        print("---------------------------------------")
        print("Testing standard LSTM on LoS dataset...")
        print("---------------------------------------")
        print("\n")
        TestLSTM().test_standard_lstm_los()


main()
