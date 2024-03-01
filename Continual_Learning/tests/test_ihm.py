import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from test_standard_lstm import TestLSTM

# ssh veselka@cse-stmi-s1.cse.tamu.edu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=int, required=True, help="Run tasks 1 through <task#>"
    )
    parser.add_argument(
        "--b",
        type=int,
        required=True,
        help="Specifies buffer size to be used in EWC and Replay",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Use replay, must specify > 1 task using --task",
    )
    parser.add_argument(
        "--ewc",
        action="store_true",
        help="Use EWC penalty, must specify > 1 task using --task",
    )

    args = parser.parse_args()

    task_list = [i for i in range(0, args.task)]
    buffer_size = args.b
    replay = args.replay if args.replay else False
    ewc_penalty = args.ewc if args.ewc else False

    print("---------------------------------------")
    print("Testing standard LSTM on IHM dataset...")
    print("---------------------------------------")
    print("\n")
    TestLSTM().test_standard_lstm_ihm(task_list, buffer_size, replay, ewc_penalty)


main()
