import numpy as np
from pathlib import Path
import datetime
import glob
from joblib import Parallel, delayed
from netCDF4 import Dataset
from aida.config import satellite_amf, satellite_opt, ctm_model, ddm_emis_model
from aida.interpolator import interpolator
import warnings
from scipy.io import savemat

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)


def _get_nc_attr(filename, var):
    # getting attributes
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    attr = {}
    for attrname in nc_fid.variables[var].ncattrs():
        attr[attrname] = getattr(nc_fid.variables[var], attrname)
    nc_fid.close()
    return attr


def _get_nc_attr_group_mopitt(fname):
    # getting attributes for mopitt
    nc_f = fname
    nc_fid = Dataset(nc_f, 'r')
    attr = {}
    for attrname in nc_fid.groups['HDFEOS'].groups['ADDITIONAL'].groups['FILE_ATTRIBUTES'].ncattrs():
        attr[attrname] = getattr(
            nc_fid.groups['HDFEOS'].groups['ADDITIONAL'].groups['FILE_ATTRIBUTES'], attrname)
    nc_fid.close()
    return attr


def _read_group_nc(filename, group, var):
    # reading nc files with a group structure
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    num_groups = len(group)
    if num_groups == 1:
        out = np.array(nc_fid.groups[group[0]].variables[var])
    elif num_groups == 2:
        out = np.array(nc_fid.groups[group[0]].groups[group[1]].variables[var])
    elif num_groups == 3:
        out = np.array(
            nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].variables[var])
    elif num_groups == 4:
        out = np.array(
            nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].groups[group[3]].variables[var])
    nc_fid.close()
    return np.squeeze(out)


def tropomi_reader_hcho(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       TROPOMI HCHO L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             tropomi_hcho [satellite_amf]: a dataclass format (see config.py)
    '''
    # hcho reader
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['PRODUCT'], 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, ['PRODUCT'], 'delta_time')), axis=1)/1000.0
    time = np.nanmean(time, axis=0)
    time = np.squeeze(time)
    time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_hcho.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'longitude').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                               'formaldehyde_tropospheric_air_mass_factor')
    # read total hcho
    vcd = _read_group_nc(fname, ['PRODUCT'],
                         'formaldehyde_tropospheric_vertical_column')
    scd = _read_group_nc(fname, ['PRODUCT'], 'formaldehyde_tropospheric_vertical_column') *\
        amf_total
    vcd = (vcd*6.02214*1e19*1e-15).astype('float16')
    scd = (scd*6.02214*1e19*1e-15).astype('float16')
    # read quality flag
    quality_flag = _read_group_nc(
        fname, ['PRODUCT'], 'qa_value').astype('float16')
    # read pressures for SWs
    tm5_a = _read_group_nc(
        fname, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_a')/100.0
    tm5_b = _read_group_nc(
        fname, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_b')
    ps = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float32')/100.0
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float32')
    if read_ak == True:
        SWs = np.zeros(
            (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        AKs = _read_group_nc(fname, [
            'PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'averaging_kernel').astype('float16')
    else:
        SWs = np.empty((1))
    # for some reason, in the HCHO product, a and b values are the center instead of the edges (unlike NO2)
    for z in range(0, 34):
        p_mid[z, :, :] = (tm5_a[z]+tm5_b[z]*ps[:, :])
        if read_ak == True:
            SWs[z, :, :] = AKs[:, :, z]*amf_total
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the precision
    uncertainty = _read_group_nc(fname, ['PRODUCT'],
                                 'formaldehyde_tropospheric_vertical_column_precision')
    uncertainty = (uncertainty*6.02214*1e19*1e-15).astype('float16')

    tropomi_hcho = satellite_amf(vcd, scd, time, np.empty((1)), latitude_center, longitude_center,
                                 [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], [], [], [], [], [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.10  # degree
        tropomi_hcho = interpolator(
            1, grid_size, tropomi_hcho, ctm_models_coordinate, flag_thresh=0.5)
    # return
    if tropomi_hcho != 0:
        return tropomi_hcho
    else:
        return None


def tropomi_reader_no2(fname: str, trop: bool, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       TROPOMI NO2 L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             trop [bool]: true for considering the tropospheric region only
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             tropomi_no2 [satellite_amf]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['PRODUCT'], 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, ['PRODUCT'], 'delta_time')), axis=0)/1000.0
    time = np.squeeze(time)
    time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_no2.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'longitude').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, ['PRODUCT'], 'air_mass_factor_total')
    # read no2
    if trop == False:
        vcd = _read_group_nc(
            fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'nitrogendioxide_total_column')
        scd = _read_group_nc(
            fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'nitrogendioxide_slant_column_density')
        # read the precision
        uncertainty = _read_group_nc(fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                                     'nitrogendioxide_total_column_precision')
    else:
        vcd = _read_group_nc(
            fname, ['PRODUCT'], 'nitrogendioxide_tropospheric_column')
        scd = vcd*_read_group_nc(
            fname, ['PRODUCT'], 'air_mass_factor_troposphere')
        # read the precision
        uncertainty = _read_group_nc(fname, ['PRODUCT'],
                                     'nitrogendioxide_tropospheric_column_precision')
    vcd = (vcd*6.02214*1e19*1e-15).astype('float16')
    scd = (scd*6.02214*1e19*1e-15).astype('float16')
    uncertainty = (uncertainty*6.02214*1e19*1e-15).astype('float16')
    # read quality flag
    quality_flag = _read_group_nc(
        fname, ['PRODUCT'], 'qa_value').astype('float16')
    # read pressures for SWs
    tm5_a = _read_group_nc(fname, ['PRODUCT'], 'tm5_constant_a')/100.0
    tm5_a = np.concatenate((tm5_a[:, 0], 0), axis=None)
    tm5_b = _read_group_nc(fname, ['PRODUCT'], 'tm5_constant_b')
    tm5_b = np.concatenate((tm5_b[:, 0], 0), axis=None)

    ps = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float32')/100.0
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        SWs = np.zeros(
            (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        AKs = _read_group_nc(fname, ['PRODUCT'],
                             'averaging_kernel').astype('float16')
    else:
        SWs = np.empty((1))
    for z in range(0, 34):
        p_mid[z, :, :] = 0.5*(tm5_a[z]+tm5_b[z]*ps[:, :] +
                              tm5_a[z+1]+tm5_b[z+1]*ps[:, :])
        if read_ak == True:
            SWs[z, :, :] = AKs[:, :, z]*amf_total
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the tropopause layer index
    if trop == True:
        trop_layer = _read_group_nc(
            fname, ['PRODUCT'], 'tm5_tropopause_layer_index')
        tropopause = np.zeros_like(trop_layer).astype('float16')
        for i in range(0, np.shape(trop_layer)[0]):
            for j in range(0, np.shape(trop_layer)[1]):
                if (trop_layer[i, j] > 0 and trop_layer[i, j] < 34):
                    tropopause[i, j] = p_mid[trop_layer[i, j], i, j]
                else:
                    tropopause[i, j] = np.nan
    else:
        tropopause = np.empty((1))
    tropomi_no2 = satellite_amf(vcd, scd, time, tropopause, latitude_center, longitude_center,
                                [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], [], [], [], [], [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.10  # degree
        tropomi_no2 = interpolator(
            1, grid_size, tropomi_no2, ctm_models_coordinate, flag_thresh=0.75)
    # return
    if tropomi_no2 != 0:
        return tropomi_no2
    else:
        return None


def omi_reader_no2(fname: str, trop: bool, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       OMI NO2 L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             trop [bool]: true for considering the tropospheric region only
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             omi_no2 [satellite_amf]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['GEOLOCATION_DATA'], 'Time')
    time = np.squeeze(np.nanmean(time))
    time = datetime.datetime(
        1993, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['GEOLOCATION_DATA'], 'Latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['GEOLOCATION_DATA'], 'Longitude').astype('float32')
    # read no2
    if trop == False:
        vcd = _read_group_nc(
            fname, ['SCIENCE_DATA'], 'ColumnAmountNO2')
        scd = _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfTrop') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop') +\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfStrat') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Strat')
        # read the precision
        uncertainty = _read_group_nc(fname, ['SCIENCE_DATA'],
                                     'ColumnAmountNO2Std')
    else:
        vcd = _read_group_nc(
            fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop')
        scd = _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfTrop') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop')
        # read the precision
        uncertainty = _read_group_nc(fname, ['SCIENCE_DATA'],
                                     'ColumnAmountNO2TropStd')
    vcd = (vcd*1e-15).astype('float16')
    scd = (scd*1e-15).astype('float16')
    uncertainty = (uncertainty*1e-15).astype('float16')
    # read quality flag
    cf_fraction = quality_flag_temp = _read_group_nc(
        fname, ['ANCILLARY_DATA'], 'CloudFraction').astype('float16')
    cf_fraction_mask = cf_fraction < 0.3
    cf_fraction_mask = np.multiply(cf_fraction_mask, 1.0).squeeze()

    train_ref = quality_flag_temp = _read_group_nc(
        fname, ['ANCILLARY_DATA'], 'TerrainReflectivity').astype('float16')
    train_ref_mask = train_ref < 0.2
    train_ref_mask = np.multiply(train_ref_mask, 1.0).squeeze()

    quality_flag_temp = _read_group_nc(
        fname, ['SCIENCE_DATA'], 'VcdQualityFlags').astype('float16')
    quality_flag = np.zeros_like(quality_flag_temp)*-100.0
    for i in range(0, np.shape(quality_flag)[0]):
        for j in range(0, np.shape(quality_flag)[1]):
            flag = '{0:08b}'.format(int(quality_flag_temp[i, j]))
            if flag[-1] == '0':
                quality_flag[i, j] = 1.0
            if flag[-1] == '1':
                if flag[-2] == '0':
                    quality_flag[i, j] = 1.0
    quality_flag = quality_flag*cf_fraction_mask*train_ref_mask
    # read pressures for SWs
    ps = _read_group_nc(fname, ['GEOLOCATION_DATA'],
                        'ScatteringWeightPressure').astype('float16')
    p_mid = np.zeros(
        (35, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        SWs = _read_group_nc(fname, ['SCIENCE_DATA'],
                             'ScatteringWeight').astype('float16')
        SWs = SWs.transpose((2, 0, 1))
    else:
        SWs = np.empty((1))
    for z in range(0, 35):
        p_mid[z, :, :] = ps[z]
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the tropopause pressure
    if trop == True:
        tropopause = _read_group_nc(
            fname, ['ANCILLARY_DATA'], 'TropopausePressure').astype('float16')
    else:
        tropopause = np.empty((1))
    # populate omi class
    omi_no2 = satellite_amf(vcd, scd, time, tropopause, latitude_center,
                            longitude_center, [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], [], [], [], [], [])
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.25  # degree
        omi_no2 = interpolator(
            1, grid_size, omi_no2, ctm_models_coordinate, flag_thresh=0.0)
    # return
    if omi_no2 != 0:
        return omi_no2
    else:
        return None


def omi_reader_hcho(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       OMI HCHO L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             omi_hcho [satellite_amf]: a dataclass format (see config.py)
    '''
    # we add "try" because some files have format issue thus unreadable
    try:
        # say which file is being read
        print("Currently reading: " + fname.split('/')[-1])
        # read time
        time = _read_group_nc(fname, ['geolocation'], 'time')
        time = np.squeeze(np.nanmean(time))
        time = datetime.datetime(
            1993, 1, 1) + datetime.timedelta(seconds=int(time))
        # read lat/lon at centers
        latitude_center = _read_group_nc(
            fname, ['geolocation'], 'latitude').astype('float32')
        longitude_center = _read_group_nc(
            fname, ['geolocation'], 'longitude').astype('float32')
        # read hcho
        vcd = _read_group_nc(
            fname, ['key_science_data'], 'column_amount')
        scd = _read_group_nc(fname, ['support_data'], 'amf') *\
            _read_group_nc(fname, ['key_science_data'], 'column_amount')
        # read the precision
        uncertainty = _read_group_nc(fname, ['key_science_data'],
                                     'column_uncertainty')
        vcd = (vcd*1e-15).astype('float16')
        scd = (scd*1e-15).astype('float16')
        uncertainty = (uncertainty*1e-15).astype('float16')
        # read quality flag
        cf_fraction = _read_group_nc(
            fname, ['support_data'], 'cloud_fraction').astype('float16')
        cf_fraction_mask = cf_fraction < 0.4
        cf_fraction_mask = np.multiply(cf_fraction_mask, 1.0).squeeze()

        quality_flag = _read_group_nc(
            fname, ['key_science_data'], 'main_data_quality_flag').astype('float16')
        quality_flag = quality_flag == 0.0
        quality_flag = np.multiply(quality_flag, 1.0).squeeze()

        quality_flag = quality_flag*cf_fraction_mask
        # read pressures for SWs
        ps = _read_group_nc(fname, ['support_data'],
                            'surface_pressure').astype('float16')
        a0 = np.array([0., 0.04804826, 6.593752, 13.1348, 19.61311, 26.09201, 32.57081, 38.98201, 45.33901, 51.69611, 58.05321, 64.36264, 70.62198, 78.83422, 89.09992, 99.36521, 109.1817, 118.9586, 128.6959, 142.91, 156.26, 169.609, 181.619,
                       193.097, 203.259, 212.15, 218.776, 223.898, 224.363, 216.865, 201.192, 176.93, 150.393, 127.837, 108.663, 92.36572, 78.51231, 56.38791, 40.17541, 28.36781, 19.7916, 9.292942, 4.076571, 1.65079, 0.6167791, 0.211349, 0.06600001, 0.01])
        b0 = np.array([1., 0.984952, 0.963406, 0.941865, 0.920387, 0.898908, 0.877429, 0.856018, 0.8346609, 0.8133039, 0.7919469, 0.7706375, 0.7493782, 0.721166, 0.6858999, 0.6506349, 0.6158184, 0.5810415, 0.5463042,
                       0.4945902, 0.4437402, 0.3928911, 0.3433811, 0.2944031, 0.2467411, 0.2003501, 0.1562241, 0.1136021, 0.06372006, 0.02801004, 0.006960025, 8.175413e-09, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        p_mid = np.zeros(
            (np.size(a0)-1, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        if read_ak == True:
            SWs = _read_group_nc(fname, ['support_data'],
                                 'scattering_weights').astype('float16')
        else:
            SWs = np.empty((1))
        for z in range(0, np.size(a0)-1):
            p_mid[z, :, :] = 0.5*((a0[z] + b0[z]*ps) + (a0[z+1] + b0[z+1]*ps))
        # remove bad SWs
        SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                     (SWs > 100.0) | (SWs < 0.0))] = 0.0
        # no need to read tropopause for hCHO
        tropopause = np.empty((1))
        # populate omi class
        omi_hcho = satellite_amf(vcd, scd, time, tropopause, latitude_center,
                                 longitude_center, [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], [], [], [], [], [])

        # interpolation
        if (ctm_models_coordinate is not None):
            print('Currently interpolating ...')
            grid_size = 0.25  # degree
            omi_hcho = interpolator(
                1, grid_size, omi_hcho, ctm_models_coordinate, flag_thresh=0.0)
        # return
        if omi_hcho != 0:
            return omi_hcho
        else:
            return None
    except:
        return None


def mopitt_reader_co(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_opt:
    '''
       MOPITT CO L3 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             mopitt_co [satellite_opt]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read timeHDFEOS/SWATHS/OMI Column Amount O3/Geolocation Fields/Latitude
    attr_mopitt = _get_nc_attr_group_mopitt(fname)
    StartTime = (attr_mopitt["StartTime"])
    EndTime = (attr_mopitt["StopTime"])

    time = 0.5*(StartTime+EndTime)
    time = datetime.datetime(
        1993, 1, 1) + datetime.timedelta(seconds=int(time))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['HDFEOS', 'GRIDS', 'MOP03',
                'Data Fields'], 'Latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['HDFEOS', 'GRIDS', 'MOP03',
                'Data Fields'], 'Longitude').astype('float32')
    longitude_center, latitude_center = np.meshgrid(
        longitude_center, latitude_center)
    longitude_center = np.transpose(longitude_center)
    latitude_center = np.transpose(latitude_center)
    # read total CO
    vcd = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                 'Data Fields'], 'RetrievedCOTotalColumnDay')
    vcd[np.where((vcd <= 0) | (np.isinf(vcd)))] = np.nan
    vcd = (vcd*1e-15).astype('float16')
    dryair_col = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                        'Data Fields'], 'DryAirColumnDay')
    x_col = (1e6*vcd/(dryair_col*1e-15)).astype('float32')
    apriori_profile = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                             'Data Fields'], 'APrioriCOMixingRatioProfileDay').transpose((2, 0, 1))
    apriori_profile[apriori_profile <= 0] = np.nan
    apriori_surface = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                             'Data Fields'], 'APrioriCOSurfaceMixingRatioDay')
    surface_pressure = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                              'Data Fields'], 'SurfacePressureDay')
    apriori_surface[apriori_surface <= 0] = np.nan
    apriori_col = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                         'Data Fields'], 'APrioriCOTotalColumnDay')
    apriori_col = (apriori_col*1e-15).astype('float16')
    apriori_col[apriori_col <= 0] = np.nan
    # read quality flag
    uncertainty = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                         'Data Fields'], 'RetrievedCOTotalColumnMeanUncertaintyDay')
    uncertainty = (uncertainty*1e-15).astype('float32')
    # no need to read tropopause for total CO
    tropopause = np.empty((1))
    # read pressures for AKs
    ps = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                'Data Fields'], 'Pressure').astype('float16')
    p_mid = np.zeros(
        (9, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        AKs = _read_group_nc(fname, ['HDFEOS', 'GRIDS', 'MOP03',
                                     'Data Fields'], 'TotalColumnAveragingKernelDay')*1e-15
        AKs = AKs.transpose((2, 0, 1)).astype('float16')
    else:
        AKs = np.empty((1))
    for z in range(0, 9):
        p_mid[z, :, :] = ps[z]

    # populate mopitt class
    mopitt = satellite_opt(vcd, time, [], tropopause, latitude_center,
                           longitude_center, [], [], uncertainty, np.ones_like(
                               vcd), p_mid, AKs, [], [], [], [],
                           apriori_col, apriori_profile, surface_pressure, apriori_surface, x_col)
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.5  # degree
        mopitt = interpolator(
            1, grid_size, mopitt, ctm_models_coordinate, flag_thresh=0.0)

    # return
    if mopitt != 0:
        return mopitt
    else:
        return None


def tropomi_reader(product_dir: str, satellite_product_name: str, ctm_models_coordinate: dict, YYYYMM: str, trop: bool, read_ak=True, num_job=1):
    '''
        reading tropomi data
             product_dir [str]: the folder containing the tropomi data
             satellite_product_name [str]: so far we support:
                                         "NO2"
                                         "HCHO"
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             trop [bool]: true for considering the tropospheric region only
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [tropomi]: the tropomi @dataclass
    '''

    # find L2 files first
    L2_files = sorted(glob.glob(product_dir + "/*" + str(YYYYMM) + "*.nc"))
    # read the files in parallel
    if satellite_product_name.split('_')[-1] == 'NO2':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(tropomi_reader_no2)(
            L2_files[k], trop, ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    elif satellite_product_name.split('_')[-1] == 'HCHO':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(tropomi_reader_hcho)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    return list(filter(lambda item: item is not None, outputs_sat))


def omi_reader(product_dir: str, satellite_product_name: str, ctm_models_coordinate: dict, YYYYMM: str, trop: bool, read_ak=True, num_job=1):
    '''
        reading omi data
             product_dir [str]: the folder containing the tropomi data
             satellite_product_name [str]: so far we support:
                                         "NO2"
                                         "HCHO"
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             trop [bool]: true for considering the tropospheric region only
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [tropomi]: the tropomi @dataclass
    '''

    # find L2 files first

    print(product_dir + "/*" + YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.nc")
    L2_files = sorted(glob.glob(product_dir + "/*" +
                                YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.nc"))
    # read the files in parallel
    if satellite_product_name.split('_')[-1] == 'NO2':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(omi_reader_no2)(
            L2_files[k], trop, ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    elif satellite_product_name.split('_')[-1] == 'HCHO':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(omi_reader_hcho)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))

    return list(filter(lambda item: item is not None, outputs_sat))


def mopitt_reader(product_dir: str, ctm_models_coordinate: dict, YYYYMM: str, read_ak=True, num_job=1):
    '''
        Reading mopitt data
             product_dir [str]: the folder containing the tropomi data
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [tropomi]: the mopitt @dataclass
    '''
    L3_files = sorted(glob.glob(product_dir + "/*" +
                                YYYYMM[0:4] + YYYYMM[4::] + "*.he5"))
    outputs_sat = Parallel(n_jobs=num_job)(delayed(mopitt_reader_co)(
        L3_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L3_files)))
    return list(filter(lambda item: item is not None, outputs_sat))


def cmaq_reader_wrapper(dir_mcip: str, dir_cmaq: str, YYYYMM: str, k: int, gasname: str):
    '''
        cmaq reader wrapper
             dir_mcip [str]: the folder containing the mcip outputs
             dir_cmaq [str]: the folder containing the cmaq conc outputs
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             k [int]: the index of the file
             gasname [str]: the name of the gas to read
        Output [ctm_model]: the ctm @dataclass
    '''
    # finding the right files
    date = datetime.datetime.strptime(str(k), '%j').date()
    cmaq_target_file = glob.glob(dir_cmaq + "/CCTM_CONC_v52*" +
                                 YYYYMM[:4] + "%03d" % int(k) + "*.nc")
    cmaq_target_file = cmaq_target_file[0]
    grd_file_2d = dir_mcip + "/GRIDCRO2D_" + \
        YYYYMM[2:4] + date.strftime('%m%d')
    met_file_2d = dir_mcip + "/METCRO2D_" + YYYYMM[2:4] + date.strftime('%m%d')
    met_file_3d = dir_mcip + "/METCRO3D_" + YYYYMM[2:4] + date.strftime('%m%d')

    print("Currently reading: " + cmaq_target_file.split('/')[-1])
    # reading time and coordinates
    lat = _read_nc(grd_file_2d, 'LAT')
    lon = _read_nc(grd_file_2d, 'LON')
    time_var = _read_nc(cmaq_target_file, 'TFLAG')
    # populating cmaq time
    time = []
    for t in range(0, np.shape(time_var)[0]):
        cmaq_date = datetime.datetime.strptime(
            str(time_var[t, 0, 0]), '%Y%j').date()
        time.append(datetime.datetime(int(cmaq_date.strftime('%Y')), int(cmaq_date.strftime('%m')),
                                      int(cmaq_date.strftime('%d')), int(time_var[t, 0, 1]/10000.0), 0, 0) +
                    datetime.timedelta(minutes=0))

    prs = _read_nc(met_file_3d, 'PRES').astype('float32')/100.0  # hPa
    surf_prs = _read_nc(met_file_2d, 'PRSFC').astype('float32')/100.0
    delp = prs.copy()
    # calculate delta pressure
    for i in range(0, np.shape(prs)[1]):
        if i == 0:  # the first layer
            delp[:, i, :, :] = 2.0*(surf_prs - prs[:, 0, :, :])
        elif i == np.shape(delp)[1]-1:  # the last layer
            delp[:, i, :, :] = prs[:, i-1, :, :] - prs[:, i, :, :]
        else:  # the between
            delp[:, i, :, :] = (prs[:, i, :, :] + prs[:, i-1, :, :]) * \
                0.5 - (prs[:, i+1, :, :] + prs[:, i, :, :])*0.5
    if gasname == 'HCHO':
        gasname = 'FORM'
    # read gas in ppbv
    gas = _read_nc(cmaq_target_file, gasname)*1000.0  # ppb
    gas = gas.astype('float32')
    # populate cmaq_data format
    cmaq_data = ctm_model(lat, lon, time, gas, prs, [], delp, 'CMAQ', False)

    return cmaq_data


def cmaq_reader_ddm_emis_wrapper(dir_ddm: str, dir_emis: str, YYYYMM: str, k: int, gasname: str, err_anthro: float,
                                 err_bio: float, err_bb: float, err_light: float, err_avi: float):
    '''
        cmaq reader DDM wrapper
             dir_ddm [str]: the folder containing the ddm outputs
             dir_emis [str]: the folder containing the emission outputs
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             k [int]: the index of the file
             gasname [str]: the name of the gas to read
             err_[x][float]: fractional errors of [anthro/bio/bb/light/avi] emission sectors
        Output [ddm_emis_model]: the ddm_emiss structure @dataclass
    '''
    # locate different files for different compounds
    if gasname == 'NO2':
        file_ddm = dir_ddm + "/CCTM_v52.exe.ASENS.v52_DDM_NOX_" + \
            YYYYMM[:4] + "%03d" % int(k)
    if gasname == 'HCHO':
        file_ddm = dir_ddm + "/CCTM_v52.exe.ASENS.v52_DDM_VOC_" + \
            YYYYMM[:4] + "%03d" % int(k)
    if gasname == 'ISOP':
        file_ddm = dir_ddm + "/CCTM_v52.exe.ASENS.v52_DDM_ISOP_" + \
            YYYYMM[:4] + "%03d" % int(k)

    file_emis_bio = dir_emis + "/CCTM_ACMAP_EMIS_beis_v52_" + \
        YYYYMM[:4] + "%03d" % int(k) + ".nc"
    file_emis_bb = dir_emis + "/CCTM_ACMAP_EMIS_fire_v52_" + \
        YYYYMM[:4] + "%03d" % int(k) + ".nc"
    file_emis_anthro = dir_emis + "/CCTM_ACMAP_EMIS_gc_v52_" + \
        YYYYMM[:4] + "%03d" % int(k) + ".nc"
    file_emis_light = dir_emis + "/CCTM_ACMAP_EMIS_LNT_v52_" + \
        YYYYMM[:4] + "%03d" % int(k) + ".nc"
    # convert tjd to yymmdd for aviation emiss
    date_reformat = datetime.datetime.strptime(YYYYMM[:4] + "_" + "%03d" % int(k), '%Y_%j')
    avi_str_date  = date_reformat.strftime('%Y%m%d')
    file_emis_avi = dir_emis + "/CMAQ_ready2_aviation_" + \
        avi_str_date[2:] + ".nc"

    print("Currently reading ddm and emis ... ")
    print(file_ddm.split('/')[-1])
    print(file_emis_bio.split('/')[-1])
    print(file_emis_bb.split('/')[-1])
    print(file_emis_anthro.split('/')[-1])

    if gasname == 'NO2':
        ddmname = 'NO2_ENX'
        emisname = ["NO", "NO2"]
        emis_mole_w = [30.0, 46.0]

    if gasname == 'HCHO':
        ddmname = 'FORM_EVM'
        emisname = ["FORM", "ETH", "ALD2", "ALDX", "ISOP", "ETOH", "PAR", "OLE", "TERP", "XYLMN",
                    "ACET", "ETHY", "IOLE", "KET", "NAPH"]
        emis_mole_w = [30.0, 28.0, 44.0, 58.1, 68.1, 46.1, 72.1,
                       42.1, 136.2, 106.2, 58.1, 26.0, 56.1, 72.1, 128.2]

    if gasname == 'ISOP':
        ddmname = 'ISOP_EIS'
        emisname = 'ISOP'
        emis_mole_w = 68.1

    #aviation emissions were produced differently so we need to specify what it has
    emis_avi_list = ["SO2","PEC","NO","NO2","FORM","ALD2","ETOH","MEOH","ETHA","PAR","OLE",
                     "ISOP","CO","POC","NH3"]
    emis_bio = []
    emis_bb = []
    emis_anthro = []
    emis_light = []
    emis_avi = []
    emis_tot = []

    ddm_out = _read_nc(file_ddm, ddmname).astype(
        'float32')*1000.0  # time: 0 ~ 23 UTC, unit: ppbv
    # timeflag for ddm
    time_var_ddm = _read_nc(file_ddm, 'TFLAG')

    for i in range(len(emisname)):
        temp = _read_nc(file_emis_bio, emisname[i]).astype(
            'float32')*emis_mole_w[i]
        emis_bio.append(temp)
        temp = _read_nc(file_emis_bb, emisname[i]).astype(
            'float32')*emis_mole_w[i]
        emis_bb.append(temp)
        temp = _read_nc(file_emis_anthro, emisname[i]).astype(
            'float32')*emis_mole_w[i]
        emis_anthro.append(temp)
        temp = _read_nc(file_emis_light, emisname[i]).astype(
            'float32')*emis_mole_w[i]
        emis_light.append(temp)
        if emisname[i] in emis_avi_list:
           temp = _read_nc(file_emis_avi, emisname[i]).astype(
               'float32')*emis_mole_w[i]
           emis_avi.append(temp)

    time_var_emis = _read_nc(file_emis_anthro, 'TFLAG')
    time_var_emis = time_var_emis[1:, :, :]

    # sum over the species
    # unit g/s, time: 0~24 UTC but always zero at 0 UTC
    # divide by 12x12 km to ease the interpolation
    emis_bio = np.sum(emis_bio, axis=0).squeeze()/12.0/12.0
    emis_bb = np.sum(emis_bb, axis=0).squeeze()/12.0/12.0
    emis_anthro = np.sum(emis_anthro, axis=0).squeeze()/12.0/12.0
    emis_light = np.sum(emis_light, axis=0).squeeze()/12.0/12.0
    emis_avi = np.sum(emis_avi, axis=0).squeeze()/12.0/12.0

    # sum over vertical distribution
    # unit g/s, time: 0~24 UTC but always zero at 0 UTC
    emis_bio = np.sum(emis_bio, axis=1).squeeze()
    emis_bb = np.sum(emis_bb, axis=1).squeeze()
    emis_anthro = np.sum(emis_anthro, axis=1).squeeze()
    emis_light = np.sum(emis_light, axis=1).squeeze()
    emis_avi = np.sum(emis_avi, axis=1).squeeze()

    emis_bio = emis_bio[1:, :, :]  # removed 0 UTC, so 1 ~ 24 UTC
    emis_bb = emis_bb[1:, :, :]
    emis_anthro = emis_anthro[1:, :, :]
    emis_light = emis_light[1:, :, :]
    emis_avi = emis_avi[1:, :, :]

    # emission for model has lightning and aviation emissions in addition to emis_tot
    emis_tot = emis_bio + emis_bb + emis_anthro + emis_light + emis_avi

    err_emis = ((emis_anthro/emis_tot)**2)*((err_anthro/100.0*emis_anthro)**2) + ((emis_bio/emis_tot)**2)*((err_bio/100.0*emis_bio)**2) + \
        ((emis_bb/emis_tot)**2)*((err_bb/100.0*emis_bb)**2) + \
        ((emis_light/emis_tot)**2)*((err_light/100.0*emis_light)**2) + \
        ((emis_avi/emis_tot)**2)*((err_avi/100.0*emis_avi)**2)

    err_emis[np.isinf(err_emis)] = 0.0
    err_emis = np.sqrt(err_emis)  # same unit as the emissions

    # time for ddm and emiss list files
    time_ddm = []
    time_emis = []
    for t in range(np.shape(time_var_ddm)[0]):
        cmaq_date = datetime.datetime.strptime(
            str(time_var_ddm[t, 0, 0]), '%Y%j').date()
        time_ddm.append(datetime.datetime(int(cmaq_date.strftime('%Y')), int(cmaq_date.strftime('%m')),
                                          int(cmaq_date.strftime('%d')), int(time_var_ddm[t, 0, 1]/10000.0), 0, 0) +
                        datetime.timedelta(minutes=0))

    for t in range(np.shape(time_var_emis)[0]):
        cmaq_date = datetime.datetime.strptime(
            str(time_var_emis[t, 0, 0]), '%Y%j').date()
        time_emis.append(datetime.datetime(int(cmaq_date.strftime('%Y')), int(cmaq_date.strftime('%m')),
                                           int(cmaq_date.strftime('%d')), int(time_var_emis[t, 0, 1]/10000.0), 0, 0) +
                         datetime.timedelta(minutes=0))

    ddm_emis = ddm_emis_model(
        time_ddm, time_emis, ddm_out, emis_tot, err_emis, False)

    return ddm_emis


def CMAQ_reader(product_dir: str, mcip_product_dir: str, ddm_product_dir: str, emis_product_dir: str, YYYYMM: str, gas_to_be_saved: str, read_inv: bool, error_frac: list):
    '''
       GMI reader
       Inputs:
             product_dir [str]: the folder containing the CMAQ data
             mcip_product_dir [str]: the folder containing the MCIP data
             ddm_product_dir [str]: the folder containing the CMAQ DDM output
             emis_product_dir [str]: the folder containing the emission data
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             gases_to_be_saved [str]: name of gases to be loaded. e.g., ['NO2']
             read_inv [str]: whether read ddm output and emissions or not
             error_frac [list]: fractional errors of anthro, bio, bb, lightning, and aviation
       Output:
             cmaq_fields [ctm_model]: a dataclass format (see config.py)
    '''
    if int(YYYYMM[:4]) % 4 != 0:
        leap_year = 0
    elif int(YYYYMM[:4]) % 400 == 0:
        leap_year = 1
    elif int(YYYYMM[:4]) % 100 == 0:
        leap_year = 0
    else:
        leap_year = 1

    if leap_year == 1:
        jday_mm_st = np.array([1, 32, 61, 92, 122, 153, 183, 214, 245,
                               275, 306, 336])
        jday_mm_ed = np.array([31, 60, 91, 121, 152, 182, 213, 244, 274,
                               305, 335, 366])
    elif leap_year == 0:
        jday_mm_st = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274,
                               305, 335])
        jday_mm_ed = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273,
                               304, 334, 365])

    target_jdays = range(
        jday_mm_st[int(YYYYMM[-2:])-1], jday_mm_ed[int(YYYYMM[-2:])-1]+1)

    outputs_ctm = []
    outputs_ddm = []

    for k in target_jdays:
        outputs_ctm.append(cmaq_reader_wrapper(
            mcip_product_dir, product_dir, YYYYMM, k, gas_to_be_saved))
        if read_inv == True:
            outputs_ddm.append(cmaq_reader_ddm_emis_wrapper(
                ddm_product_dir, emis_product_dir, YYYYMM, k, gas_to_be_saved, error_frac[0], error_frac[1], error_frac[2], error_frac[3], error_frac[4]))

    return outputs_ctm, outputs_ddm


class readers(object):

    def __init__(self) -> None:
        pass

    def add_satellite_data(self, product_name: str, product_dir: Path):
        '''
            add L2 data
            Input:
                product_name [str]: a string specifying the type of data to read:
                                   TROPOMI_NO2
                                   TROPOMI_HCHO
                                   TROPOMI_CH4
                                   TROPOMI_CO
                                   OMI_NO2
                                   OMI_HCHO
                                   OMI_O3
                                   MOPITT
                                   GOSAT
                product_dir  [Path]: a path object describing the path of L2 files
        '''
        self.satellite_product_dir = product_dir
        self.satellite_product_name = product_name

    def add_ctm_data(self, product_name: int, product_dir: Path, mcip_dir: Path,
                     ddm_dir: Path, emis_dir: Path):
        '''
            add CTM data
            Input:
                product_name [str]: an string specifying the type of data to read:
                                "CMAQ"
                product_dir  [Path]: a path object describing the path of CTM files
                mcip_dir  [Path]: a path object describing the path of MCIP files
                ddm_dir [Path]: a path object describing the path of DDM output
                emis dir [Path]: a path object describing the path of emission files (beis, fire, gc, tot)

        '''

        self.ctm_product_dir = product_dir
        self.mcip_product_dir = mcip_dir
        self.ctm_product = product_name
        self.ddm_dir = ddm_dir
        self.emis_product_dir = emis_dir

    def read_satellite_data(self, YYYYMM: str, read_ak=True, trop=False, num_job=1):
        '''
            read L2 satellite data
            Input:
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             trop[bool]: true for only including the tropospheric region (relevant for NO2 only)
             num_job [int]: the number of jobs for parallel computation
        '''
        satellite = self.satellite_product_name.split('_')[0]
        ctm_models_coordinate = {}
        ctm_models_coordinate["Latitude"] = self.ctm_data[0].latitude
        ctm_models_coordinate["Longitude"] = self.ctm_data[0].longitude
        if satellite == 'TROPOMI':
            self.sat_data = tropomi_reader(self.satellite_product_dir.as_posix(),
                                           self.satellite_product_name, ctm_models_coordinate,
                                           YYYYMM,  trop, read_ak=read_ak, num_job=num_job)
        elif satellite == 'OMI':
            self.sat_data = omi_reader(self.satellite_product_dir.as_posix(),
                                       self.satellite_product_name, ctm_models_coordinate,
                                       YYYYMM,  trop, read_ak=read_ak, num_job=num_job)
        elif satellite == 'MOPITT':
            self.sat_data = mopitt_reader(self.satellite_product_dir.as_posix(),
                                          ctm_models_coordinate,
                                          YYYYMM, read_ak=read_ak, num_job=num_job)
        else:
            raise Exception("the satellite is not supported, come tomorrow!")

    def read_ctm_data(self, YYYYMM: str, gas: str, error_frac: list, read_ddm=False, averaged=False):
        '''
            read ctm data
            Input:
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             gas [str]: name of the gas to be loaded. e.g., 'NO2'
             error_frac [list]: fractional errors of anthro, bio, and bb
             read_ddm [bool]: whether read ddm output and emissions or not
             averaged [bool]: averaging ctm over a month or use the daily values
        '''
        self.read_ddm = read_ddm

        ctm_data = CMAQ_reader(self.ctm_product_dir.as_posix(),
                               self.mcip_product_dir.as_posix(),
                               self.ddm_dir.as_posix(),
                               self.emis_product_dir.as_posix(),
                               YYYYMM, gas, read_ddm, error_frac)

        self.ctm_data = ctm_data[0]
        self.ddm_data = ctm_data[1]
        ctm_data = []


# testing
if __name__ == "__main__":
    reader_obj = readers()
    reader_obj.add_ctm_data('CMAQ', Path('/nobackup/jjung13/ACMAP_CMAQ_OUT/BASE/BC_monthly_add_CO'),
                            Path('/nobackup/jjung13/ACMAP_mcipout/2019'),
                            Path('/nobackup/jjung13/ACMAP_CMAQ_OUT/DDM/2019_NOX'),
                            Path('/nobackup/jjung13/ACMAP_CMAQ_OUT/DDM/2019_VOC'),
                            Path('/nobackup/jjung13/ACMAP_CMAQ_OUT/DDM/2019_ISOP'),
                            Path('/nobackup/jjung13/ACMAP_4D_EMIS/2019/sources'))

    reader_obj.read_ctm_data('201904', 'HCHO', read_inv=True, averaged=True)
 #   reader_obj.add_satellite_data(
 #       'TROPOMI_NO2',Path('/nobackup/jjung13/ACMAP_satellite/TROPOMI_NO2/'))

 #   reader_obj.read_satellite_data(
 #       '201905', read_ak=False, num_job=18)
# %%
#    latitude = reader_obj.ctm_data[0].latitude
#    longitude = reader_obj.ctm_data[0].longitude
#    output = np.zeros((np.shape(latitude)[0], np.shape(
#        latitude)[1], len(reader_obj.sat_data)))
#    output2 = np.zeros_like(output)
#    counter = -1
#    for trop in reader_obj.sat_data:
#        counter = counter + 1
#        if trop is None:
#            continue
#        output[:, :, counter] = trop.vcd
#        #output2[:, :, counter] = trop.ctm_xcol

#    moutput = {}
#    moutput["sat"] = output
#    moutput["lat"] = latitude
#    moutput["lon"] = longitude
#    savemat("vcds_mopitt.mat", moutput)
