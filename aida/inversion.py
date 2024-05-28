import numpy as np
from scipy.io import savemat
from kneed import KneeLocator
from aida.config import OI_result


def inversion(Xa: np.array, Y: np.array, Sa: np.array, So: np.array, regularization_on=True):
    ''' 
    Inversion with CMAQ and satellite..
    '''
    print('Inversion with working...!')


