import os
import numpy as np

folder_path = "./Diff_10/PositionDiff"

sums = np.zeros(4)

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            floats = [float(value) for value in file.readline().split()[:4]]
            sums += floats

print("Sum of the first 4 floats from all files:")
print(sums)

# Case 13 : [ 124.   13.    8.    0.]
# 