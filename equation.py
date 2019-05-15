import numpy as np

s_dim = 10  # some number
i_dim = 20  # some number
d_dim = 2   # some number

# Q for s
Q = np.matrix([[1 for _ in range(d_dim)] for _ in range(s_dim)])
# P for i
P = np.matrix([[1 for _ in range(d_dim)] for _ in range(i_dim)])
# Bp for s
b_p = np.array([1 for _ in range(s_dim)])
# Bq for i
b_q = np.array([1 for _ in range(i_dim)])
# Bp for s
x_s = np.array([1 for _ in range(s_dim)])
# Bq for i
x_i = np.array([1 for _ in range(i_dim)])

def predict_single(x_s, x_i):
    y1 = np.dot(x_s, Q)
    y2 = np.dot(x_i, P)
    y = np.dot(y1, y2.T)
    dots = np.dot(x_s, b_p) + np.dot(x_i, b_q)
    y += dots
    return y.item()

print(predict_single(x_s, x_i))
