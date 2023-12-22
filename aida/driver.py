from aida.reader import readers
from pathlib import Path
from aida.amf_recal import amf_recal
from aida.averaging import averaging
from aida.optimal_interpolation import OI
from aida.report import report
from aida.ak_conv import ak_conv
import numpy as np
from scipy.io import savemat
from numpy import dtype
from netCDF4 import Dataset
import os


class aida(object):

    def __init__(self) -> None:
        pass

    def read_data(self, ctm_type: str, ctm_path: Path, mcip_path: Path, ctm_gas_name: str,
                  sat_type: str, sat_path: Path, YYYYMM: str, averaged=False, read_ak=True, trop=False, num_job=1):
        reader_obj = readers()
        reader_obj.add_ctm_data(ctm_type, ctm_path, mcip_path)
        reader_obj.read_ctm_data(YYYYMM, ctm_gas_name, averaged=averaged, num_job=num_job)
        reader_obj.add_satellite_data(
            sat_type, sat_path)
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.reader_obj = reader_obj
        self.gasname = ctm_gas_name[0]
        reader_obj = []

    def recal_amf(self):

        self.reader_obj.sat_data = amf_recal(
            self.reader_obj.ctm_data, self.reader_obj.sat_data)

    def conv_ak(self):

        self.reader_obj.sat_data = ak_conv(
            self.reader_obj.ctm_data, self.reader_obj.sat_data)

    def average(self, startdate: str, enddate: str, gasname=None):
        '''
            average the data
            Input:
                startdate [str]: starting date in YYYY-mm-dd format string
                enddate [str]: ending date in YYYY-mm-dd format string  
        '''
        self.sat_averaged_vcd, self.sat_averaged_error, self.ctm_averaged_vcd, self.aux1, self.aux2 = averaging(
            startdate, enddate, self.reader_obj)

    def oi(self, error_ctm=50.0):

        self.ctm_averaged_vcd_corrected, self.ak_OI, self.increment_OI, self.error_OI = OI(self.ctm_averaged_vcd, self.sat_averaged_vcd,
                                                                                           (self.ctm_averaged_vcd*error_ctm/100.0)**2, self.sat_averaged_error**2, regularization_on=True)

    def reporting(self, fname: str, gasname, folder='report'):

        # pick the right latitude and longitude
        # the right one is the coarsest one so
        if np.size(self.reader_obj.ctm_data[0].latitude)*np.size(self.reader_obj.ctm_data[0].longitude) > \
           np.size(self.reader_obj.sat_data[0].latitude_center)*np.size(self.reader_obj.sat_data[0].longitude_center):

            lat = self.reader_obj.sat_data[0].latitude_center
            lon = self.reader_obj.sat_data[0].longitude_center
        else:
            lat = self.reader_obj.ctm_data[0].latitude
            lon = self.reader_obj.ctm_data[0].longitude

        report(lon, lat, self.ctm_averaged_vcd, self.ctm_averaged_vcd_corrected,
               self.sat_averaged_vcd, self.sat_averaged_error, self.increment_OI, self.ak_OI, self.error_OI, self.aux1, self.aux2, fname, folder, gasname)

    def write_to_nc(self, output_file, output_folder='diag'):
        ''' 
        Write the final results to a netcdf
        ARGS:
            output_file (char): the name of file to be outputted
        '''
        # writing
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        ncfile = Dataset(output_folder + '/' + output_file + '.nc', 'w')

        # create the x and y dimensions.
        ncfile.createDimension('x', np.shape(self.sat_averaged_vcd)[0])
        ncfile.createDimension('y', np.shape(self.sat_averaged_vcd)[1])

        data1 = ncfile.createVariable(
            'sat_averaged_vcd', dtype('float32').char, ('x', 'y'))
        data1[:, :] = self.sat_averaged_vcd

        data2 = ncfile.createVariable(
            'ctm_averaged_vcd_prior', dtype('float32').char, ('x', 'y'))
        data2[:, :] = self.ctm_averaged_vcd

        data3 = ncfile.createVariable(
            'ctm_averaged_vcd_posterior', dtype('float32').char, ('x', 'y'))
        data3[:, :] = self.ctm_averaged_vcd_corrected

        data4 = ncfile.createVariable(
            'sat_averaged_error', dtype('float32').char, ('x', 'y'))
        data4[:, :] = self.sat_averaged_error

        data5 = ncfile.createVariable(
            'ak_OI', dtype('float32').char, ('x', 'y'))
        data5[:, :] = self.ak_OI

        data6 = ncfile.createVariable(
            'error_OI', dtype('float32').char, ('x', 'y'))
        data6[:, :] = self.error_OI

        scaling_factor = self.ctm_averaged_vcd_corrected/self.ctm_averaged_vcd
        scaling_factor[np.where((np.isnan(scaling_factor)) | (np.isinf(scaling_factor)) |
                       (scaling_factor == 0.0))] = 1.0
        data7 = ncfile.createVariable(
            'scaling_factor', dtype('float32').char, ('x', 'y'))
        data7[:, :] = scaling_factor

        data8 = ncfile.createVariable(
            'lon', dtype('float32').char, ('x', 'y'))
        data8[:, :] = self.reader_obj.sat_data[0].longitude_center

        data9 = ncfile.createVariable(
            'lat', dtype('float32').char, ('x', 'y'))
        data9[:, :] = self.reader_obj.sat_data[0].latitude_center

        data10 = ncfile.createVariable(
            'aux1', dtype('float32').char, ('x', 'y'))
        data10[:, :] = self.aux1

        data11 = ncfile.createVariable(
            'aux2', dtype('float32').char, ('x', 'y'))
        data11[:, :] = self.aux2

        ncfile.close()


# testing
if __name__ == "__main__":

    aida_obj = aida()
    aida_obj.read_data('CMAQ', Path('/nobackup/jjung13/ACMAP_CMAQ_OUT/BASE/BC_monthly/'),
                           Path('/nobackup/jjung13/ACMAP_mcipout/2019/'), 'NO2', 'OMI_NO2',
                           Path('/nobackup/asouri/GITS/AIDA/aida/download_bucket/omi_no2/'), '201905',
                           averaged=True, read_ak=True, trop=True, num_job=12)
    aida_obj.recal_amf()
    #aida_obj.conv_ak()
    aida_obj.average('2019-05-01', '2019-06-01')
    aida_obj.oi(error_ctm=50.0)
    aida_obj.reporting('OMI_NO2_new', 'NO2', folder='report')
    #aida_obj.write_to_nc('NO2_200503_new', 'diag')
