from numpy import array
import numpy as np

PATH_TO_NEW_RESULTS = "."

GRID = [[None] * 8 for _ in range(21)]
j = -2
for task in ["ihm", "phen", "dec", "los"]:
    mimic_prime_val_arr = []
    mimic_second_val_arr = []
    mimic_prime_std_arr = []
    mimic_second_std_arr = []
    i = 1
    j += 2
    # print(bs)
    for region in ["south", "midwest", "west", "northeast"]:
        bst = 0
        loc = i, j
        for method in ["base", "ewc", "trrep", "rep", "comb"]:
            bs = 3500 if (task == "dec" or task == "los") else 500
            bs = 0 if method == "base" else bs
            print(bs)
            try:
                with open(
                    f"{PATH_TO_NEW_RESULTS}/{task}/{region}/{task}_{bs}_{method}_CE.txt",
                    "r",
                ) as file:
                    primary_metric = True
                    res = [
                        0,
                        0,
                        0,
                        0,
                    ]  # [prime val, prime std, secondary val, secondary std]

                    for line in file:
                        if "Per Task Average:" in line and "Best" not in line:
                            arr = eval(line.split(":")[1])  # eval the array part
                            res[0] = arr[1][0]
                            res[2] = arr[1][1]

                            mimic_prime_val_arr.append(arr[0][0])
                            mimic_second_val_arr.append(arr[0][1])

                        if "Std Dev:" in line:
                            arr = eval("[" + line.split("[")[1])  # eval the array part

                            if primary_metric:
                                primary_metric = False
                                res[1] = arr[1]
                                mimic_prime_std_arr.append(arr[0])
                            else:
                                res[3] = arr[1]
                                mimic_second_std_arr.append(arr[0])
                    # print(i)
                    GRID[i][j] = f"{res[0]:.3f} ({res[1]:.3f})"
                    GRID[i][j + 1] = f"{res[2]:.3f} ({res[3]:.3f})"

                    if res[0] >= bst:
                        bst = res[0]
                        loc = i, j

                    i += 1
            except FileNotFoundError as e:
                GRID[i][j] = f"0.000 (0.000)"
                GRID[i][j + 1] = f"0.000 (0.000)"
                i += 1

        curr = GRID[loc[0]][loc[1]]
        vl, std = curr.split(" ")
        GRID[loc[0]][loc[1]] = "\\textbf{" + vl + "} " + std

    if len(mimic_prime_val_arr) == 0:
        mimic_prime_val_arr = [0]
    if len(mimic_second_val_arr) == 0:
        mimic_second_val_arr = [0]

    GRID[0][
        j
    ] = f"{np.mean(mimic_prime_val_arr):.3f} ({np.std(mimic_prime_val_arr):.3f})"
    GRID[0][
        j + 1
    ] = f"{np.mean(mimic_second_val_arr):.3f} ({np.std(mimic_second_val_arr):.3f})"

REGIONS = ["MIMIC-III", "South", "Midwest", "West", "Northeast"]
METHOD = ["Baseline", "EWC", "Replay", "Adj Replay", "Combined"]
track_region = 1
for i, row in enumerate(GRID):
    m = METHOD[(i - 1) % 5]
    if i == 0:
        print("& \\multirow{1}{*}{MIMIC-III}", end=" & ")
    elif (i - 1) % 5 == 0:
        print(m + " & \\multirow{5}{*}{" + REGIONS[track_region] + "}", end=" & ")
        track_region += 1
    else:
        print(m, end=" & & ")

    for j, cell in enumerate(row):
        suffix = "\\midrule" if (i - 1) % 5 == 4 else ""
        if j == len(row) - 1:
            print(cell, end=f" \\\\ {suffix}")
        else:
            print(cell, end=" & ")
    print()
