import numpy as np
import pandas as pd

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

# I mean dafuw is rho?
def rho(x): return x

# loss function
# @ param
#   pred_scores: list of predicted scores | [float]
#   pos_score: the score of the positive(actually chosen) item | float
#   N_s: number of samples | int=25
# @ return
#   the loss for a session x_s
def BPR(pred_scores, pos_score, N_s=25):
    #   in paper        log(rho()) but what is rho
    return (1/(N_s))*sum((log(rho(pos_score - pred_scores[j])) for i in range(N_s)))

# other loss function
# @ param
#   pred_scores: list of predicted scores | [float]
#   pos_score: the score of the positive(actually chosen) item | float
#   N_s: number of samples | int=25
# @ return
#   the loss for a session x_s
def TOP1(pred_scores, pos_score, N_s=25):
    return (1/(N_s))*sum(rho(pred_scores[j] - pos_score) + rho(pred_scores[j]**2) for i in range(N_s)))


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
