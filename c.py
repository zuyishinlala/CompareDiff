import numpy as np

# Sample arrays
A = np.array([[1, 2, 3, 4, 5, 6],
              [7, 8, 9, 10, 11, 12]])

B = np.array([13, 14])

# Append B at the end of A along axis 1
result = np.concatenate((A, np.expand_dims(B, axis=1)), axis=1)

print(result)
