from aida.reader import readers
from pathlib import Path
from aida.amf_recal import amf_recal
from aida.averaging import averaging
from aida.optimal_interpolation import OI
from aida.inversion import inv_sat, inv_sat_aqs
from aida.report import report
from aida.ak_conv import ak_conv
import numpy as np
from scipy.io import savemat
from numpy import dtype
from netCDF4 import Dataset
import os
import glob


class aida(object):

    def __init__(self) -> None:

        self.do_run_OI = False
        self.do_run_inversion = False
        self.oi_result = []
        self.inversion_result = []
        self.X1 = 0.0
        self.first_iteration = True

    def read_data(self, ctm_type: str, ctm_path: Path, mcip_path: Path, ctm_gas_name: str,
                  sat_type: str, sat_path: Path, YYYYMM: str, error_fraction: list,
                  ddm_path=[], emis_path=[], read_ddm=False, averaged=False, read_ak=True, trop=True, num_job=1
                  ):
        reader_obj = readers()
        reader_obj.add_ctm_data(
            ctm_type, ctm_path, mcip_path, ddm_path, emis_path)
        reader_obj.read_ctm_data(
            YYYYMM, ctm_gas_name, error_fraction, read_ddm=read_ddm, averaged=averaged)
        reader_obj.add_satellite_data(
            sat_type, sat_path)
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.reader_obj = reader_obj
        self.gasname = ctm_gas_name[0]
        reader_obj = []
        self.YYYYMM = YYYYMM

    def recal_amf(self):

        if self.reader_obj.read_ddm == False:
            self.reader_obj.sat_data = amf_recal(
                self.reader_obj.ctm_data, self.reader_obj.sat_data, [], ddm_read=False)
        else:
            self.reader_obj.sat_data = amf_recal(
                self.reader_obj.ctm_data, self.reader_obj.sat_data, self.reader_obj.ddm_data, ddm_read=True)

    def conv_ak(self):

        self.reader_obj.sat_data = ak_conv(
            self.reader_obj.ctm_data, self.reader_obj.sat_data)

    def average(self, startdate: str, enddate: str):
        '''
            average the data
            Input:
                startdate [str]: starting date in YYYY-mm-dd format string
                enddate [str]: ending date in YYYY-mm-dd format string
        '''
        self.averaged_fields = averaging(
            startdate, enddate, self.reader_obj)

    def bias_correct(self, sat_type, gasname):
        # apply bias correction based on several validation studies

        if sat_type == "TROPOMI" and gasname == "NO2":
            print("applying the bias correction for TROPOMI NO2")
            sat_averaged_vcd_bias_corrected = (
                self.averaged_fields.sat_vcd - 0.32)/0.66
            '''
            reference: Amir
            '''
        elif sat_type == "TROPOMI" and gasname == "HCHO":
            print("applying the bias correction for TROPOMI HCHO")
            sat_averaged_vcd_bias_corrected = (
                self.averaged_fields.sat_vcd - 0.90)/0.59
            '''
            reference: Amir
            '''
        elif sat_type == "OMI" and gasname == "NO2":
            print("applying the bias correction for OMI NO2")
            '''
            need to work on these again
            '''
            sat_averaged_vcd_bias_corrected = (
                self.averaged_fields.sat_vcd - 0.32)/0.63
            '''
            reference: Johnson et al., 2023 -- offset is from TROPOMI NO2, slope is from Matt's paper
            '''
        elif sat_type == "OMI" and gasname == "HCHO":
            print("applying the bias correction for OMI HCHO")
            sat_averaged_vcd_bias_corrected = (
                self.averaged_fields.sat_vcd - 0.821)/(0.79)
            '''
            reference: Ayazpour et al., Submitted, Auto Ozone Monitoring Instrument (OMI) Collection 4 Formaldehyde Product
	    based on Figure 11, monthly climatology regression
            '''

        else:
            print("NOT applying the bias correction for satellite VCDs")
            sat_averaged_vcd_bias_corrected = self.averaged_fields.sat_vcd

        # populating the averaged vcds with the bias corrected ones
        self.averaged_fields.sat_vcd = sat_averaged_vcd_bias_corrected

    def oi(self, error_ctm=50.0):

        self.do_run_OI = True
        self.oi_result = OI(self.averaged_fields.ctm_vcd, self.averaged_fields.sat_vcd,
                            (self.averaged_fields.ctm_vcd*error_ctm/100.0)**2, self.averaged_fields.sat_err**2, regularization_on=True)

    def emission_last_iter(self, inv_folder: str, YYYYMM: str, gasname):

        self.first_iteration = False
        inverse_file = (glob.glob(inv_folder + "/" + gasname + "*" + str(YYYYMM) + ".nc"))
        print("reading the previous inversion from " + str(inverse_file[0]))
        if not inverse_file:
            raise Exception(
                "We don't have the previous inversion states")
        nc_fid = Dataset(inverse_file[0], 'r')
        self.X1 = np.array(nc_fid.variables['inv_posterior_emissions'])
        nc_fid.close()

    def inversion(self, gasname, sat_type: str, inv_type: str, aqs_folder=None):
        self.do_run_inversion = True
        if (inv_type == 'SAT+AQS') and (gasname == 'HCHO'):
            inv_type == 'SAT'  # we haven't implemented it plus surface HCHO obs are sparse

        if inv_type == 'SAT':
            self.inversion_result = inv_sat(self.averaged_fields.sat_vcd, self.averaged_fields.sat_err**2,
                                            self.averaged_fields.ctm_vcd, self.averaged_fields.ddm_vcd /
                                            self.averaged_fields.emis_total,
                                            self.averaged_fields.emis_total, self.X1,
                                            self.averaged_fields.emis_error**2, self.first_iteration, gasname, sat_type, regularization_on=True)
        if inv_type == 'SAT+AQS':
            # read AQS data here
            file_aqs = sorted(glob.glob(aqs_folder + "/*" +
                                        self.YYYYMM + "*.csv"))
            output_aqs = np.loadtxt(file_aqs[0], delimiter=',')
            # prepare 2D AQS output for the current lat/lon maps
            if np.size(self.reader_obj.ctm_data[0].latitude)*np.size(self.reader_obj.ctm_data[0].longitude) > \
               np.size(self.reader_obj.sat_data[0].latitude_center)*np.size(self.reader_obj.sat_data[0].longitude_center):

                lat = self.reader_obj.sat_data[0].latitude_center
                lon = self.reader_obj.sat_data[0].longitude_center
            else:
                lat = self.reader_obj.ctm_data[0].latitude
                lon = self.reader_obj.ctm_data[0].longitude

            AQS_map = np.zeros_like(lat)
            for i in range(0, np.shape(lat)[0]):
                for j in range(0, np.shape(lat)[1]):
                    cost = np.sqrt((lat[i, j]-output_aqs[:, 0])
                                   ** 2+(lon[i, j]-output_aqs[:, 1])**2)
                    index_i = np.argwhere(cost <= 0.05)
                    Z = output_aqs[:, 2]
                    chosen_aqs = Z[index_i]
                    if np.size(chosen_aqs) > 1:
                        chosen_aqs = np.nanmean(chosen_aqs)
                    if np.size(chosen_aqs) != 0:
                        AQS_map[i, j] = chosen_aqs
            # save everything for testing
            #output_test = {}
            #output_test["sat_vcd"] = self.averaged_fields.sat_vcd
            #output_test["AQS_map"] = AQS_map
            #output_test["ctm_vcd"] = self.averaged_fields.ctm_vcd
            #output_test["ctm_surf"] = self.averaged_fields.ctm_surface
            #output_test["emis"] = self.averaged_fields.emis_total
            #output_test["K_vcd"] = self.averaged_fields.ddm_vcd/self.averaged_fields.emis_total
            #output_test["K_surf"] = self.averaged_fields.ddm_surface/self.averaged_fields.emis_total
            #output_test["Se"] = self.averaged_fields.emis_error**2
            #output_test["So"] = self.averaged_fields.sat_err**2
            #output_test["X1"] = self.X1
            #savemat("test_second_iteration.mat", output_test)
            self.inversion_result = inv_sat_aqs(self.averaged_fields.sat_vcd, AQS_map, self.averaged_fields.sat_err**2,
                                                self.averaged_fields.ctm_vcd, self.averaged_fields.ctm_surface, self.averaged_fields.ddm_vcd /
                                                self.averaged_fields.emis_total, self.averaged_fields.ddm_surface /
                                                self.averaged_fields.emis_total, self.averaged_fields.emis_total, self.X1,
                                                self.averaged_fields.emis_error**2, self.first_iteration, gasname, sat_type,
                                                aqs_error_percent=20.0, regularization_on=True)

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

        report(lon, lat, self.averaged_fields, self.oi_result,
               self.inversion_result, fname, folder, gasname, read_ddm=self.reader_obj.read_ddm)

    def write_to_nc(self, output_file, output_folder='diag', read_ddm=False):
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
        ncfile.createDimension('x', np.shape(self.averaged_fields.sat_vcd)[0])
        ncfile.createDimension('y', np.shape(self.averaged_fields.sat_vcd)[1])
        ncfile.createDimension('t', None)  # unlimited
        ncfile.createDimension('u', None)  # unlimited
        # generic fields
        data1 = ncfile.createVariable(
            'sat_averaged_vcd', dtype('float32').char, ('x', 'y'))
        data1[:, :] = self.averaged_fields.sat_vcd

        data2 = ncfile.createVariable(
            'ctm_averaged_vcd_prior', dtype('float32').char, ('x', 'y'))
        data2[:, :] = self.averaged_fields.ctm_vcd

        data4 = ncfile.createVariable(
            'sat_averaged_error', dtype('float32').char, ('x', 'y'))
        data4[:, :] = self.averaged_fields.sat_err

        if np.size(self.reader_obj.ctm_data[0].latitude)*np.size(self.reader_obj.ctm_data[0].longitude) > \
           np.size(self.reader_obj.sat_data[0].latitude_center)*np.size(self.reader_obj.sat_data[0].longitude_center):

            lat = self.reader_obj.sat_data[0].latitude_center
            lon = self.reader_obj.sat_data[0].longitude_center
        else:
            lat = self.reader_obj.ctm_data[0].latitude
            lon = self.reader_obj.ctm_data[0].longitude

        data8 = ncfile.createVariable(
            'lon', dtype('float32').char, ('x', 'y'))
        data8[:, :] = lon

        data9 = ncfile.createVariable(
            'lat', dtype('float32').char, ('x', 'y'))
        data9[:, :] = lat

        data10 = ncfile.createVariable(
            'aux1', dtype('float32').char, ('x', 'y'))
        data10[:, :] = self.averaged_fields.aux1

        data11 = ncfile.createVariable(
            'aux2', dtype('float32').char, ('x', 'y'))
        data11[:, :] = self.averaged_fields.aux2

        # DDM
        if read_ddm == True:
            data12 = ncfile.createVariable(
                'ddm_vcd', dtype('float32').char, ('x', 'y'))
            data12[:, :] = self.averaged_fields.ddm_vcd

            data13 = ncfile.createVariable(
                'emis_tot', dtype('float32').char, ('x', 'y'))
            data13[:, :] = self.averaged_fields.emis_total

            data14 = ncfile.createVariable(
                'emis_err', dtype('float32').char, ('x', 'y'))
            data14[:, :] = self.averaged_fields.emis_error

        if self.oi_result:
            # OI results
            data3 = ncfile.createVariable(
                'ctm_averaged_vcd_posterior', dtype('float32').char, ('x', 'y'))
            data3[:, :] = self.oi_result.ctm_corrected

            data5 = ncfile.createVariable(
                'ak_OI', dtype('float32').char, ('x', 'y'))
            data5[:, :] = self.oi_result.ak

            data6 = ncfile.createVariable(
                'error_OI', dtype('float32').char, ('x', 'y'))
            data6[:, :] = self.oi_result.error_analysis

            scaling_factor = self.oi_result.ctm_corrected/self.averaged_fields.ctm_vcd
            scaling_factor[np.where((np.isnan(scaling_factor)) | (np.isinf(scaling_factor)) |
                                    (scaling_factor == 0.0))] = 1.0
            data7 = ncfile.createVariable(
                'scaling_factor_OI', dtype('float32').char, ('x', 'y'))
            data7[:, :] = scaling_factor

        if self.inversion_result:
            # inversion results
            data17 = ncfile.createVariable(
                'inv_posterior_emissions', dtype('float32').char, ('x', 'y'))
            data17[:, :] = self.inversion_result.post_emis

            data18 = ncfile.createVariable(
                'inv_ak', dtype('float32').char, ('x', 'y'))
            data18[:, :] = self.inversion_result.ak

            data19 = ncfile.createVariable(
                'inv_increment', dtype('float32').char, ('x', 'y'))
            data19[:, :] = self.inversion_result.increment

            data20 = ncfile.createVariable(
                'inv_error_post', dtype('float32').char, ('x', 'y'))
            data20[:, :] = self.inversion_result.error_analysis

            data21 = ncfile.createVariable(
                'inv_ratio_post_prior', dtype('float32').char, ('x', 'y'))
            data21[:, :] = self.inversion_result.ratio

        data15 = ncfile.createVariable(
            'gap', dtype('float32').char, ('t', 'x', 'y'))
        data15[:, :, :] = self.averaged_fields.gap_field

        data16 = ncfile.createVariable(
            'time', dtype('float64').char, ('u'))
        data16[:] = self.averaged_fields.time_sat

        ncfile.close()

    def savedaily(self, folder, gasname, date):
        # extract sat data
        if not os.path.exists(folder):
            os.makedirs(folder)
        latitude = self.reader_obj.sat_data[0].ctm_lat
        longitude = self.reader_obj.sat_data[0].ctm_lon
        vcd_sat = np.zeros((np.shape(latitude)[0], np.shape(
            latitude)[1], len(self.reader_obj.sat_data)))
        vcd_err = np.zeros_like(vcd_sat)
        vcd_ctm = np.zeros_like(vcd_sat)
        time_sat = np.zeros((len(self.reader_obj.sat_data)))
        counter = -1
        for sat in self.reader_obj.sat_data:
            counter = counter + 1
            if sat is None:
                continue
            vcd_sat[:, :, counter] = sat.vcd
            vcd_ctm[:, :, counter] = sat.ctm_vcd
            vcd_err[:, :, counter] = sat.uncertainty
            time_sat[counter] = 10000.0*sat.time.year + 100.0 * \
                sat.time.month + sat.time.day + sat.time.hour/24.0

        sat = {"vcd_sat": vcd_sat, "vcd_ctm": vcd_ctm,
               "vcd_err": vcd_err, "time_sat": time_sat, "lat": latitude, "lon": longitude}
        savemat(folder + "/" + "sat_data_" +
                gasname + "_" + date + ".mat", sat)


# testing
if __name__ == "__main__":

    aida_obj = aida()
    errors = [0.5, 1.0, 3.0]
    aida_obj.read_data('CMAQ', Path('/nobackup/jjung13/ACMAP_CMAQ_OUT/BASE/BC_monthly/'),
                       Path('/nobackup/jjung13/ACMAP_mcipout/2019/'), 'NO2', 'OMI_NO2',
                       Path(
                           '/nobackup/asouri/GITS/AIDA/aida/download_bucket/omi_no2/'), '201905',
                       errors, read_ddm=True, averaged=True, read_ak=True, trop=True, num_job=12)
    aida_obj.recal_amf()
    # aida_obj.conv_ak()
    aida_obj.average('2019-05-01', '2019-06-01')
    aida_obj.oi(error_ctm=50.0)
    aida_obj.reporting('OMI_NO2_new', 'NO2', folder='report')
    #aida_obj.write_to_nc('NO2_200503_new', 'diag')
