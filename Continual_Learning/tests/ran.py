import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)

arrays = []
# Example data: a list of 3 5x5x2 arrays
nparr1 = np.random.rand(3, 3, 2)
nparr2 = np.random.rand(3, 3, 2)
nparr3 = np.random.rand(3, 3, 2)
arrays.append(nparr1)
arrays.append(nparr2)
arrays.append(nparr3)

averages = np.mean(arrays, axis=0)

first_values = averages[:, :, 0]
second_values = averages[:, :, 1]
m = averages.shape[0]

tasks = [f"Task {i}" for i in range(1, m + 1)]
df1 = pd.DataFrame(first_values, tasks, tasks)
df2 = pd.DataFrame(second_values, tasks, tasks)

# Define the class labels (replace with your actual class labels)

# Plot confusion matrix for the first value of the pair
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.heatmap(df1, annot=True, cmap="Blues", fmt=".3f")
plt.yticks(rotation=0)
plt.xlabel("Trained")
plt.ylabel("Evaluated")
plt.gca().xaxis.set_ticks_position("top")
plt.gca().xaxis.set_label_position("top")
plt.title("Trained vs. Evaluated", pad=20)


plt.subplot(1, 2, 2)
sns.heatmap(df2, annot=True, cmap="Blues", fmt=".3f")
plt.yticks(rotation=0)
plt.xlabel("Trained")
plt.ylabel("Evaluated")
plt.gca().xaxis.set_ticks_position("top")
plt.gca().xaxis.set_label_position("top")
plt.title("Trained vs. Evaluated", pad=20)

# plt.tight_layout()
plt.savefig("results/ex.png")
