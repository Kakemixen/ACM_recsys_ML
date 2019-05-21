import numpy as np
import pandas as pd
from math import sqrt

### algorithm relevant
# the singel prediction equation presented in the paper
# @ param
#   x_s: session vector | [float]
#   x_i: item vector attributes one-hot + price | [float]
# @ return
#   the score for item x_i in session x_s | float
def predict_single(x_s, x_i):
    y1 = np.dot(x_s, Q)
    y2 = np.dot(x_i, P)
    y = np.dot(y1, y2.T)
    dots = np.dot(x_s, b_p) + np.dot(x_i, b_q)
    y += dots
    return y.item()

# sigma = sigmoid
def sigma(x): return x

# for both loss functions
# @ param
#   pred_scores: list of predicted scores | [float]
#   pos_score: the score of the positive(actually chosen) item | float
#   N_s: number of samples | int=25
# @ return
#   the loss for a session x_s

# a loss function
def BPR(pred_scores, pos_score, N_s=25):
    return (1/(N_s))*sum((log(sigma(pos_score - pred_scores[j])) for i in range(N_s)))

# another loss function
def TOP1(pred_scores, pos_score, N_s=25):
    return (1/(N_s))*sum(sigma(pred_scores[j] - pos_score) + sigma(pred_scores[j]**2) for i in range(N_s))

# an attempt will be made
def bregman(func, x, y):
    return func(x) - func(y) # - np.dot(part_diff(func(y)), x - y)

# @ param
#   alpha: learning rate | float in (0,1)
#   T: number of trials | int > 0
# @ return
#   the optimal x (point in parameter space)
def ADAGRAD(alpha, T):
    print("not implemended ADAGRAD")
    # for each trial:
    for i in range(T):
        # suffer loss
        # recieve subgradient
        # update g-matrix
        # set new H_t and Psi_t
        # H_t = deltaI + diag(s_t)
        # Psi_t(x) = 1/2 * np.dot(x, H_t*x) : x | columns vector

        # updates:
        # 1
        # x_{t+1} = argmin_x()
        # 2
        # x_{t+1} = argmin_x()
        pass

def ADAGRAD_update(params, G_diag, grad_t, learn_rate=0.1):
    eps = 0.0000001
    return np.array([ param[i] - (learn_rate/sqrt(eps * G_diag[i])) * grad[i]  \
            for i in range(len(params))])

# @ param
#   g_matrix: a matrix of column vectors | np.matrix
def calc_G_diag(g_matrix):
    return np.array([0 for i in range(g_matrix)])








# getting data relevant
data_folder = "../data/"

# @ param
#   itemIDs: arrays of ID | [str]
# @ return
#   items: a matrix where rows are the item vectors | np.matrix
def get_item_vector(itemIDs):
    path = data_folder+"item_metadata.csv"


# @ param
#   sessionID: single ID | str
#   chunk: chunk to look for ID in | pd.DataFrame
#   chunk_next: next chunk to use if ID is at end of chunk | pd.DataFrame
# @ return
#   session: the session in an independen DF | pd.DataFrame
def get_session(sessionID, chunk=None, chunk_next=None):
    path = data_folder+"test.csv"


#testing of functions
if __name__ == "__main__":
    #initialize data_dimensions
    s_dim = 10  # some number
    i_dim = 20  # some number
    d_dim = 2   # some number


    #initialize data in said dimensions as arrays of 1's
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

    print(predict_single(x_s, x_i))
