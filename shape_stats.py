import os
import pandas as pd
import glob

# Define the directory path
directory_path = "WUPERR_CLP/data"

subdirectories = ["Hosp-A", "Hosp-B", "Hosp-C", "Hosp-D"]

# Initialize an empty list to store DataFrames
dfs = []
tot = 0
tot_not_165 = 0

# Loop through each subdirectory
for subdirectory in subdirectories:
    sub_dfs = []

    count_not_165 = 0
    total_files = 0

    print("Working on subdirectory: ", subdirectory)
    subdirectory_path = os.path.join(directory_path, subdirectory)

    csv_files = glob.glob(os.path.join(subdirectory_path, "*.csv"))

    for csv_file in csv_files:
        total_files += 1
        df = pd.read_csv(csv_file, header=None)
        if df.shape[1] != 165:
            count_not_165 += 1

        sub_dfs.append(df)
        # print values in this column if the column has at least one 1
        # if df.iloc[:, 142].eq(1).any():
        #     print(df.iloc[:, 142])

    print(f"Total files in {subdirectory}: {total_files}")
    print(f"Number of instances with != 165 columns: {count_not_165}")

    tot += total_files
    tot_not_165 += count_not_165

    print("Concatenating DataFrames in subdirectory: ", subdirectory)
    sub_df = pd.concat(sub_dfs)
    print("shape bef: ", sub_df.shape)
    df = sub_df.dropna(axis=1, how="all")
    print("Seperating features and labels...")

    # take everything before and after 142nd index
    features = pd.concat([df.iloc[:, :142], df.iloc[:, 143:]], axis=1)
    labels = df.iloc[:, 142]

    print("shape after: ", df.shape)

    dfs.append(df)
    print("-------------------")


# Concatenate all DataFrames in the list along the rows axis
print("Concatenating all DataFrames")
result_df = pd.concat(dfs)
result_df = result_df.dropna(axis=1, how="all")
print("shape: ", result_df.shape)
print(tot)

print("% = ", tot_not_165 / tot * 100)
