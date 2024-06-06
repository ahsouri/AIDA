import numpy as np
from scipy.io import savemat
from kneed import KneeLocator
from aida.config import inversion_result


def IV(Y: np.array, So: np.array, So_sys: np.array, F: np.array, K: np.array, X0: np.array, Sa: np.array, index_iteration:int, gasname: str, sat_type: str, regularization_on=True):
    
    ''' 
    Inversion with CMAQ and satellite..
    G = SaK^T(KSaK^T + So)^-1
    if index_iteration == 0
        X_i+1 = X_0 + G (Y_0 - F_0)
    else
        X_i+1 = X_0 + G (Y_i - F_i + K_i *(X_i - X_0))
    Xb: posteriori emissions
    X_0: priori emission
    Sa: Error cov of emission
    So: Error cov of satellite
    Y: satellite VCD
    F: modeled VCD
    K: jacobian

    Input:
        index_iteration [int]: indicating it's the first iteration or not 
    '''
    print('Inversion is working...!, index_iteration :', index_iteration, 'gasname :',gasname, ', sat_type: ',sat_type)

    # adding fixed error into So##
    if sat_type == "TROPOMI" and gasname == "NO2":
        print("finding So for TROPOMI NO2")
        So_new = So + 2.4**2
        '''
        reference: Geffen et al., 2022, Sentinel-5P TROPOMI NO2 retrieval: impact of version v2.2 improvements and comparisons with OMI and
        ground-based data
        '''
    elif sat_type == "TROPOMI" and gasname == "HCHO":
        print("finding So for TROPOMI HCHO")
        So_new = So + So_sys
    elif sat_type == "OMI" and gasname == "NO2":
        print("finding So for OMI NO2")
        So_new = So + 4.1**2
        '''
        reference: Johnson et al., 2023
        '''
    elif sat_type == "OMI" and gasname == "HCHO":
        print("finding So for OMI HCHO")
        So_new = So + 3.59**2
        '''
        reference: Ayazpour et al., submitted, Aura Ozone Monitoring Instrument (OMI) Collection 4 Formaldehyde Product
        '''
    ##############################
    Y[Y < 0] = 0.0
    if regularization_on == True:
        scaling_factors = np.arange(0.1, 10, 0.1)
        scaling_factors = list(scaling_factors)
    else:
        scaling_factors = []
        scaling_factors.append(1.0)

    averaging_kernel_mean = []
    kalman_gain = []
    Sb = []
    averaging_kernel = []
    Sa_new = []
    for reg in scaling_factors:
        Sa_new = Sa*float(reg)
        kalman_gain_tmp = (Sa_new*K*(K*Sa_new*K+So_new)**(-1))
        kalman_gain.append(kalman_gain_tmp)
        Sb_tmp = (np.ones_like(kalman_gain_tmp)-kalman_gain_tmp*K)*Sa_new
        Sb.append(Sb_tmp)
        AK = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
        averaging_kernel.append(AK)
        averaging_kernel_mean.append(np.nanmean(AK.flatten()))

    if regularization_on == True:
        averaging_kernel_mean = np.array(averaging_kernel_mean)
        kneedle = KneeLocator(np.array(scaling_factors),
                              averaging_kernel_mean, direction='increasing')
        knee_index = np.argwhere(np.array(scaling_factors) == kneedle.knee)
        if np.size(knee_index) == 0:
            knee_index = [0]
    else:
        knee_index = [0]

    kalman_gain = kalman_gain[int(knee_index[0])]
    averaging_kernel = averaging_kernel[int(knee_index[0])]
    Sb = Sb[int(knee_index[0])]

    if index_iteration == 0:
        increment = kalman_gain*(Y-F)
    else:
        #need to be done this part 
        increment = kalman_gain*(Y-F + K*(X1-X0))

    Xb = X0 + increment

    ratio = np.ones_like(X0)
    ratio = Xb/X0
    ratio[np.isnan(ratio)] = 1.0
    ratio[np.isinf(ratio)] = 1.0
    ratio[ratio<=0] = 1.0
    

    output = inversion_result(Xb, averaging_kernel, increment, np.sqrt(Sb), ratio)
    return output



