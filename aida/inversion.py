import numpy as np
from kneed import KneeLocator
from aida.config import inversion_result
import copy
from scipy.io import loadmat, savemat


def inv_sat(Y: np.array, So: np.array, F: np.array, K: np.array, X0: np.array,
            X1: np.array, Sa: np.array, first_iteration: bool, gasname: str, sat_type: str,
            regularization_on=True):
    ''' 
    Inversion using analytical non-linear Gauss-Newton method (satellite_only)
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
        first_iteration [bool]: indicating if it's the first iteration or not 
    '''
    print('The inversion is being executed for ',
          'gasname :', gasname, ', sat_type: ', sat_type)
    Y_new = copy.deepcopy(Y)
    Y_new[Y_new < 0] = 0.0
    if regularization_on == True:
        scaling_factors = np.arange(0.2, 20, 0.2)
        scaling_factors = list(scaling_factors)
    else:
        scaling_factors = []
        scaling_factors.append(1.0)

    averaging_kernel_mean = []
    kalman_gain = []
    Sb = []
    averaging_kernel = []
    Sa_new = []
    # note that this inversion is applied to each individual grid box separately, so the transpose of K and K
    # is identical. Likewise, we won't need to use np.matmul because each grid has one Y and one X.

    # finding the optimal reg factor
    for reg in scaling_factors:
        Sa_new = Sa*float(reg)
        kalman_gain_tmp = (Sa_new*K*(K*Sa_new*K+So)**(-1))
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
            knee_index = 4  # equal to 1.0
    else:
        knee_index = 4

    print('The optimal regularization factor is ' +
          str(scaling_factors[int(knee_index)]))

    kalman_gain = kalman_gain[int(knee_index)]
    averaging_kernel = averaging_kernel[int(knee_index)]
    Sb = Sb[int(knee_index)]

    if first_iteration == True:
        increment = kalman_gain*(Y_new-F)
    else:
        increment = kalman_gain*(Y_new-F + K*(X1-X0))

    Xb = X0 + increment

    ratio = Xb/X0
    ratio[np.isnan(ratio)] = 1.0
    ratio[np.isinf(ratio)] = 1.0
    ratio[ratio <= 0] = 1.0

    output = inversion_result(Xb, averaging_kernel,
                              increment, np.sqrt(Sb), ratio)

    return output


def inv_sat_aqs(Y: np.array, aqs_data: np.array, So: np.array, F_VCD: np.array, F_surf: np.array, K_vcd: np.array, K_surf: np.array, X0: np.array, X1: np.array,
                Sa: np.array, first_iteration: bool, gasname: str, sat_type: str, aqs_error_percent=20.0, regularization_on=True):
    ''' 
    Inversion using analytical non-linear Gauss-Newton method (satellite+aqs)

    Input:
       Xb: posteriori emissions
       X0: priori emission (Xa)
       X1: the most recent iteration (Xi) (X1=X0 for the first iteration)
       Sa: Error cov of emission
       So: Error cov of satellite
       Y: satellite VCD
       aqs_data: AQS 2D map
       F_VCD: modeled VCD
       F_surf: modeled surface conc
       K_vcd: jacobian for VCD
       k_surf: jacobian for surface conc
       first_iteration [bool]: indicating if it's the first iteration or not 
       gasname: NO2 or HCHO
       sat_type: TROPOMI or OMI or OMPS (for error augmuntation)
       aqs_error_percent: AQS errors in %
       regularization_on: the switch for using regularization

    '''
    print('The inversion is being executed for',
          'gasname :', gasname, ', sat_type: ', sat_type)

    Y_new = copy.deepcopy(Y)
    Y_new[Y_new < 0] = 0.0

    if regularization_on == True:
        scaling_factors = np.arange(0.2, 20, 0.2)
        scaling_factors = list(scaling_factors)
    else:
        scaling_factors = []
        scaling_factors.append(1.0)

    averaging_kernel_mean = []
    # finding the optimal reg factor
    for reg in scaling_factors:
        averaging_kernel = []
        for i in range(0, np.shape(Y_new)[0]):
            for j in range(0, np.shape(Y_new)[1]):
                if aqs_data[i, j] != 0.0:  # we have surface obs
                    OBS = np.array([[Y_new[i, j]], [aqs_data[i, j]]])
                    # 10% error for AQS
                    So_new = np.array(
                        [[So[i, j], 0], [0, (aqs_data[i, j]*aqs_error_percent/100.0)**2]])
                    K = np.array([[K_vcd[i, j]], [K_surf[i, j]]])
                    Sa_new = Sa[i, j]*float(reg)
                    kalman_gain_tmp = np.matmul(
                        Sa_new*K.transpose(), np.linalg.inv(np.matmul(K*Sa_new, K.transpose())+So_new))
                    Sb_tmp = (1-np.matmul(kalman_gain_tmp, K))*Sa_new
                    AK = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                    averaging_kernel.append(AK[0])
                else:  # we don't have surface obs
                    OBS = np.array([Y_new[i, j]])
                    So_new = np.array(
                        [So[i, j]])
                    K = np.array([K_vcd[i, j]])
                    Sa_new = Sa[i, j]*float(reg)
                    kalman_gain_tmp = (
                        Sa_new*K*(K*Sa_new*K+So_new)**(-1))
                    Sb_tmp = (1-kalman_gain_tmp*K)*Sa_new
                    AK = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                    averaging_kernel.append(AK)

        averaging_kernel_mean.append(np.nanmean(
            np.array(averaging_kernel).flatten()))

    if regularization_on == True:
        averaging_kernel_mean = np.array(averaging_kernel_mean)
        kneedle = KneeLocator(np.array(scaling_factors),
                              averaging_kernel_mean, direction='increasing')
        knee_index = np.argwhere(np.array(scaling_factors) == kneedle.knee)
        if np.size(knee_index) == 0:
            knee_index = 4  # equal to 1.0
    else:
        knee_index = 4

    print("knee_index")
    print(knee_index)
    print('The optimal regularization factor is ' +
          str(scaling_factors[int(knee_index)]))

    # run the inversion again using the optimal reg factor
    averaging_kernel = np.zeros_like(Y)
    increment = np.zeros_like(Y)
    Sb = np.zeros_like(Y)
    for i in range(0, np.shape(Y_new)[0]):
        for j in range(0, np.shape(Y_new)[1]):
            if aqs_data[i, j] != 0.0:  # we have surface obs
                OBS = np.array([[Y_new[i, j]], [aqs_data[i, j]]])
                So_new = np.array(
                    [[So[i, j], 0], [0, (aqs_data[i, j]*aqs_error_percent/100.0)**2]])
                K = np.array([[K_vcd[i, j]], [K_surf[i, j]]])
                Sa_new = Sa[i, j]*float(scaling_factors[int(knee_index)])
                kalman_gain_tmp = np.matmul(
                    Sa_new*K.transpose(), np.linalg.inv(np.matmul(K*Sa_new, K.transpose())+So_new))
                Sb_tmp = (1-np.matmul(kalman_gain_tmp, K))*Sa_new
                AK = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                F = np.array([[F_VCD[i, j]], [F_surf[i, j]]])

                averaging_kernel[i, j] = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                Sb[i, j] = Sb_tmp
                if first_iteration == True:
                    increment[i, j] = np.matmul(kalman_gain_tmp, (OBS-F))
                else:
                    increment[i, j] = np.matmul(
                        kalman_gain_tmp, (OBS-F+K*(X1[i, j]-X0[i, j])))

            else:  # we don't have surface obs
                OBS = np.array([Y_new[i, j]])
                So_new = np.array(
                    [So[i, j]])
                K = np.array([K_vcd[i, j]])
                Sa_new = Sa[i, j]*float(scaling_factors[int(knee_index)])
                kalman_gain_tmp = (Sa_new*K*(K*Sa_new*K+So_new)**(-1))
                Sb_tmp = (1-kalman_gain_tmp*K)*Sa_new
                Sb[i, j] = Sb_tmp
                averaging_kernel[i, j] = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                F = np.array([F_VCD[i, j]])
                if first_iteration == True:
                    increment[i, j] = kalman_gain_tmp*(OBS-F)
                else:
                    increment[i, j] = kalman_gain_tmp * \
                        (OBS-F + K*(X1[i, j]-X0[i, j]))

    Xb = X0 + increment
    ratio = Xb/X0
    ratio[np.isnan(ratio)] = 1.0
    ratio[np.isinf(ratio)] = 1.0
    ratio[ratio <= 0] = 1.0

    output = inversion_result(Xb, averaging_kernel,
                              increment, np.sqrt(Sb), ratio)

    return output


def inv_sat_aqs_dual(Y: np.array, aqs_data: np.array, So: np.array, F_VCD: np.array, F_surf: np.array, K_vcd: np.array, K_surf: np.array, X0: np.array, X1: np.array,
                     Sa: np.array, first_iteration: bool, gasname: str, sat_type: str, aqs_error_percent=20.0, regularization_on=True):
    ''' 
    Inversion using analytical non-linear Gauss-Newton method (satellite+aqs+dual)
    "Dual" means we constrain two sets of emissions (>750 hPa and <750 hPa) using two sets of observations (Sat+AQS)

    Input:
       Xb: posteriori emissions
       X0: priori emission (Xa)
       X1: the most recent iteration (Xi) (X1=X0 for the first iteration)
       Sa: Error cov of emission
       So: Error cov of satellite
       Y: satellite VCD
       aqs_data: AQS 2D map
       F_VCD: modeled VCD
       F_surf: modeled surface conc
       K_vcd: jacobian for VCD
       k_surf: jacobian for surface conc
       first_iteration [bool]: indicating if it's the first iteration or not 
       gasname: NO2 or HCHO
       sat_type: TROPOMI or OMI or OMPS (for error augmuntation)
       aqs_error_percent: AQS errors in %
       regularization_on: the switch for using regularization

    '''
    print('The inversion is being executed for',
          'gasname :', gasname, ', sat_type: ', sat_type)

    Y_new = copy.deepcopy(Y)
    Y_new[Y_new < 0] = 0.0

    if regularization_on == True:
        scaling_factors = np.arange(0.2, 20, 0.2)
        scaling_factors = list(scaling_factors)
    else:
        scaling_factors = []
        scaling_factors.append(1.0)

    averaging_kernel_mean = []
    # finding the optimal reg factor
    for reg in scaling_factors:
        averaging_kernel = []
        for i in range(0, np.shape(Y_new)[0]):
            for j in range(0, np.shape(Y_new)[1]):
                if aqs_data[i, j] != 0.0:  # we have surface obs
                    OBS = np.array([[Y_new[i, j]], [aqs_data[i, j]]])
                    # 10% error for AQS
                    So_new = np.array(
                        [[So[i, j], 0], [0, (aqs_data[i, j]*aqs_error_percent/100.0)**2]])
                    K = np.array([[K_vcd[i, j, 0], K_vcd[i, j, 1]], [
                                 K_surf[i, j, 0], K_surf[i, j, 1]]])
                    Sa_new = np.array(
                        [[Sa[i, j, 0]*float(reg), 0], [0, Sa[i, j, 1]*float(reg)]])
                    kalman_gain_tmp = np.matmul(
                        np.matmul(Sa_new, K.transpose()), np.linalg.inv(np.matmul(np.matmul(K, Sa_new), K.transpose())+So_new))
                    Sb_tmp = np.matmul(np.ones_like(
                        Sb_tmp)-np.matmul(kalman_gain_tmp, K), Sa_new)
                    AK = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                    averaging_kernel.append(np.trace(AK))
                else:  # we don't have surface obs
                    OBS = np.array([Y_new[i, j]])
                    So_new = np.array(
                        [So[i, j]])
                    K = np.array([K_vcd[i, j, 0], K_vcd[i, j, 1]])
                    Sa_new = np.array(
                        [[Sa[i, j, 0]*float(reg), 0], [0, Sa[i, j, 1]*float(reg)]])
                    kalman_gain_tmp = np.matmul(Sa_new, K.transpose()) *\
                        (np.matmul(np.matmul(K, Sa_new), K.transpose())+So_new)**(-1)
                    Sb_tmp = np.matmul(np.ones_like(
                        Sa_new)-np.matmul(kalman_gain_tmp, K), Sa_new)
                    AK = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                    averaging_kernel.append(np.trace(AK))

        averaging_kernel_mean.append(np.nanmean(
            np.array(averaging_kernel).flatten()))

    if regularization_on == True:
        averaging_kernel_mean = np.array(averaging_kernel_mean)
        kneedle = KneeLocator(np.array(scaling_factors),
                              averaging_kernel_mean, direction='increasing')
        knee_index = np.argwhere(np.array(scaling_factors) == kneedle.knee)
        if np.size(knee_index) == 0:
            knee_index = 4  # equal to 1.0
    else:
        knee_index = 4

    print("knee_index")
    print(knee_index)
    print('The optimal regularization factor is ' +
          str(scaling_factors[int(knee_index)]))

    # run the inversion again using the optimal reg factor
    shape = Y.shape + (2,)
    averaging_kernel = np.full(shape, np.nan)
    increment = np.full(shape, np.nan)
    Sb = np.full(shape, np.nan)
    for i in range(0, np.shape(Y_new)[0]):
        for j in range(0, np.shape(Y_new)[1]):
            if aqs_data[i, j] != 0.0:  # we have surface obs
                OBS = np.array([[Y_new[i, j]], [aqs_data[i, j]]])
                So_new = np.array(
                    [[So[i, j], 0], [0, (aqs_data[i, j]*aqs_error_percent/100.0)**2]])
                K = np.array([[K_vcd[i, j, 0], K_vcd[i, j, 1]], [
                             K_surf[i, j, 0], K_surf[i, j, 1]]])
                Sa_new = np.array(
                    [[Sa[i, j, 0], 0], [0, Sa[i, j, 1]]])*scaling_factors[int(knee_index)]
                kalman_gain_tmp = np.matmul(
                    np.matmul(Sa_new, K.transpose()), np.linalg.inv(np.matmul(np.matmul(K, Sa_new), K.transpose())+So_new))
                Sb_tmp = np.matmul(np.ones_like(Sa_new) -
                                   np.matmul(kalman_gain_tmp, K), Sa_new)
                F = np.array([[F_VCD[i, j]], [F_surf[i, j]]])
                averaging_kernel_temp = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                averaging_kernel[i, j, :] = np.diag(averaging_kernel_temp)
                #print(f"averagingkernel in AQS {averaging_kernel[i,j,:]}")
                Sb[i, j, :] = np.diag(Sb_tmp)
                if first_iteration == True:
                    increment[i, j, :] = np.matmul(
                        kalman_gain_tmp, (OBS-F)).squeeze()
                else:
                    increment[i, j, :] = np.matmul(
                        kalman_gain_tmp, (OBS-F+K*(X1[i, j, :]-X0[i, j, :]))).squeeze()

            else:  # we don't have surface obs
                OBS = np.array([Y_new[i, j]])
                So_new = np.array(
                    [So[i, j]])
                K = np.array([[K_vcd[i, j, 0]],
                              [K_vcd[i, j, 1]]]).transpose()
                Sa_new = np.array(
                    [[Sa[i, j, 0], 0], [0, Sa[i, j, 1]]])*scaling_factors[int(knee_index)]
                kalman_gain_tmp = np.matmul(Sa_new, K.transpose()) *\
                    (np.matmul(np.matmul(K, Sa_new), K.transpose())+So_new)**(-1)
                Sb_tmp = np.matmul(np.ones_like(Sa_new) -
                                   np.matmul(kalman_gain_tmp, K), Sa_new)
                Sb[i, j, :] = np.diag(Sb_tmp)
                averaging_kernel_temp = np.ones_like(Sb_tmp)-(Sb_tmp)/(Sa_new)
                averaging_kernel[i, j, :] = np.diag(averaging_kernel_temp)
                #print(f"averagingkernel in SAT {averaging_kernel[i,j,:]}")
                F = np.array([F_VCD[i, j]])
                if first_iteration == True:
                    increment[i, j, :] = kalman_gain_tmp.squeeze()*(OBS-F)
                else:
                    increment[i, j, :] = kalman_gain_tmp.squeeze() *\
                        (OBS-F + K*(X1[i, j, :]-X0[i, j, :]))

    Xb = X0 + increment
    ratio = Xb/X0
    ratio[np.isnan(ratio)] = 1.0
    ratio[np.isinf(ratio)] = 1.0
    ratio[ratio <= 0] = 1.0

    output = inversion_result(Xb, averaging_kernel,
                              increment, np.sqrt(Sb), ratio)

    return output


# testing
if __name__ == "__main__":
    matvar = loadmat(
        '/home/asouri/git_repos/mule4/inversion_aqs/test_second_iteration.mat')
    AQS_map = matvar["AQS_map"]
    output = inv_sat_aqs_dual(matvar["sat_vcd"], AQS_map, matvar["So"],
                              matvar["ctm_vcd"], matvar["ctm_surf"], matvar["K_vcd"], matvar["K_surf"],
                              matvar["emis"], matvar["X1"],
                              matvar["Se"], True, 'NO2', 'TROPOMI',
                              regularization_on=True)
    outputs = {}
    outputs["Xb"] = output.post_emis
    outputs["AK"] = output.ak
    outputs["increment"] = output.increment
    outputs["Sb"] = output.error_analysis
    outputs["ratio"] = output.ratio
    savemat("test_dual.mat", outputs)
