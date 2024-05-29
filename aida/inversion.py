import numpy as np
from scipy.io import savemat
from kneed import KneeLocator
from aida.config import inversion_result


def IV(Y: np.array, So: np.array, F: np.array, K: np.array, X: np.array, Sa: np.array, index_iteration: int, regularization_on=True):
    ''' 
    Inversion with CMAQ and satellite..
    G = SaK^T(KSaK^T + So)^-1
    if index_iteration == 0
        X_i+1 = X_0 + G (Y_0 - F_0)
    else
        X_i+1 = X_0 + G (Y_i - F_i + K_i *(X_i - X_0))
    X_i+1: posteriori emissions
    X_0: priori emission
    Sa: Error cov of emission
    So: Error cov of satellite
    Y: satellite VCD
    F: modeled VCD
    K: jacobian


        XXb = Xa + SaK^T(KSaK^T + So)^-1 * (Y-K*Xa)
    Input:
        index_iteration [int]: indicating it's the first iteration or not 
    '''
    print('Inversion is working...!, index_iteration :', index_iteration)
    print(np.shape(Y))
    print(np.shape(So))
    print(np.shape(F))
    print(np.shape(K))
    print(np.shape(X))
    print(np.shape(Sa))







