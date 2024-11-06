import numpy as np
import datetime
from aida.config import satellite_amf, averaged_field


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)

def remove_non_numbers(lst):
    return [x for x in lst if isinstance(x, (int, float))]

def error_averager(error_X: np.array):
    error_Y = np.zeros((np.shape(error_X)[1],np.shape(error_X)[2]))*np.nan
    for i in range(0,np.shape(error_X)[1]):
        for j in range(0,np.shape(error_X)[2]):
            temp = []
            for k in range(0,np.shape(error_X)[0]):
                temp.append(error_X[k,i,j])
            temp = remove_non_numbers(temp)
            temp = np.array(temp)
            error_Y[i,j] = np.sum(temp)/(np.size(temp)**2)

    error_Y = np.sqrt(error_Y)
    return error_Y

def averaging(startdate: str, enddate: str, reader_obj):
    '''
          average the data
          Input:
              startdate [str]: starting date in YYYY-mm-dd format string
              enddate [str]: ending date in YYYY-mm-dd format string
              gasname [str], bias_sat (True or False), and sat_type (TROPOMI, OMI..) 
                are for applying bias correction for satellite VCD
    '''
    # convert dates to datetime
    start_date = datetime.date(int(startdate[0:4]), int(
        startdate[5:7]), int(startdate[8:10]))
    end_date = datetime.date(int(enddate[0:4]), int(
        enddate[5:7]), int(enddate[8:10]))
    list_days = []
    list_months = []
    list_years = []
    for single_date in _daterange(start_date, end_date):
        list_days.append(single_date.day)
        list_months.append(single_date.month)
        list_years.append(single_date.year)

    list_days = np.array(list_days)
    list_months = np.array(list_months)
    list_years = np.array(list_years)

    sat_averaged_vcd = np.zeros((np.shape(reader_obj.sat_data[0].latitude_center)[0],
                                 np.shape(reader_obj.sat_data[0].latitude_center)[
        1],
        len(range(np.min(list_months),
                  np.max(list_months)+1)),
        len(range(np.min(list_years), np.max(list_years)+1))))

    #sat_samples = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_averaged_error = np.zeros_like(sat_averaged_vcd)*np.nan
    ctm_averaged_vcd = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_aux1 = np.zeros_like(sat_averaged_vcd)*np.nan
    sat_aux2 = np.zeros_like(sat_averaged_vcd)*np.nan
    if reader_obj.read_ddm == True:
        emis_averaged = np.zeros_like(sat_averaged_vcd)*np.nan
        ddm_averaged = np.zeros_like(sat_averaged_vcd)*np.nan
        ddm_surface_averaged = np.zeros_like(sat_averaged_vcd)*np.nan
        ctm_surface_averaged = np.zeros_like(sat_averaged_vcd)*np.nan
        emis_err_averaged = np.zeros_like(sat_averaged_vcd)*np.nan
    for year in range(np.min(list_years), np.max(list_years)+1):
        for month in range(np.min(list_months), np.max(list_months)+1):
            sat_chosen_vcd = []
            sat_chosen_aux1 = []
            sat_chosen_aux2 = []
            sat_chosen_error = []
            ctm_chosen_vcd = []
            time_chosen = []
            gap_chosen = []
            if reader_obj.read_ddm == True:
                emis_chosen = []
                ddm_chosen = []
                emis_err_chosen = []
                ddm_surface_chosen = []
                ctm_chosen_surface = []
            counter = 0
            for sat_data in reader_obj.sat_data:
                if (sat_data is None):
                    continue
                counter = counter + 1
                time_sat = sat_data.time
                time_chosen.append(10000.0*sat_data.time.year + 100.0 *
                                   sat_data.time.month + sat_data.time.day + sat_data.time.hour/24.0)
                # see if it falls
                if ((time_sat.year == year) and (time_sat.month == month)):
                    gap_chosen.append(~np.isnan(sat_data.vcd))
                    sat_chosen_vcd.append(sat_data.vcd)
                    sat_chosen_error.append(sat_data.uncertainty)
                    ctm_chosen_vcd.append(sat_data.ctm_vcd)
                    if reader_obj.read_ddm == True:
                        emis_chosen.append(sat_data.emis_tot)
                        ddm_chosen.append(sat_data.ddm_vcd)
                        emis_err_chosen.append(sat_data.emis_err)
                        ddm_surface_chosen.append(sat_data.ddm_surface)
                        ctm_chosen_surface.append(sat_data.ctm_surface_conc)
                    if isinstance(sat_data, satellite_amf):
                        sat_chosen_aux1.append(sat_data.new_amf)
                        sat_chosen_aux2.append(sat_data.old_amf)
                    else:
                        sat_chosen_aux1.append(sat_data.x_col)
                        sat_chosen_aux2.append(sat_data.ctm_xcol)

            sat_chosen_vcd = np.array(sat_chosen_vcd)
            sat_chosen_vcd[np.isinf(sat_chosen_vcd)] = np.nan
            sat_chosen_error = np.array(sat_chosen_error)
            ctm_chosen_vcd = np.array(ctm_chosen_vcd)
            sat_chosen_aux1 = np.array(sat_chosen_aux1)
            sat_chosen_aux2 = np.array(sat_chosen_aux2)
            gap_chosen = np.array(gap_chosen)
            time_chosen = np.array(time_chosen, dtype=np.float64)
        if np.size(sat_chosen_vcd) != 0:
            sat_averaged_vcd[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(sat_chosen_vcd, axis=0))
            sat_averaged_error[:, :, month - min(list_months), year - min(
                list_years)] = error_averager(sat_chosen_error**2)
            ctm_averaged_vcd[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(ctm_chosen_vcd, axis=0))

            if reader_obj.read_ddm == True:
                emis_chosen = np.array(emis_chosen)
                emis_err_chosen = np.array(emis_err_chosen)
                ddm_chosen = np.array(ddm_chosen)
                ddm_surface_chosen = np.array(ddm_surface_chosen)
                ctm_chosen_surface = np.array(ctm_chosen_surface)
                emis_averaged[:, :, month - min(list_months), year - min(
                    list_years)] = np.squeeze(np.nanmean(emis_chosen, axis=0))
                emis_err_averaged[:, :, month - min(list_months), year - min(
                    list_years)] = error_averager(emis_err_chosen**2)
                ddm_averaged[:, :, month - min(list_months), year - min(
                    list_years)] = np.squeeze(np.nanmean(ddm_chosen, axis=0))
                ddm_surface_averaged[:, :, month - min(list_months), year - min(
                    list_years)] = np.squeeze(np.nanmean(ddm_surface_chosen, axis=0))
                ctm_surface_averaged[:, :, month - min(list_months), year - min(
                    list_years)] = np.squeeze(np.nanmean(ctm_chosen_surface, axis=0))
        if np.size(sat_chosen_aux1) != 0:
            sat_aux1[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(sat_chosen_aux1, axis=0))
            sat_aux2[:, :, month - min(list_months), year - min(
                list_years)] = np.squeeze(np.nanmean(sat_chosen_aux2, axis=0))
    # squeeze it
    sat_averaged_vcd = sat_averaged_vcd.squeeze()
    sat_averaged_error = sat_averaged_error.squeeze()
    ctm_averaged_vcd = ctm_averaged_vcd.squeeze()
    sat_aux1 = sat_aux1.squeeze()
    sat_aux2 = sat_aux2.squeeze()
    if reader_obj.read_ddm == True:
        emis_averaged = emis_averaged.squeeze()
        emis_err_averaged = emis_err_averaged.squeeze()
        ddm_averaged = ddm_averaged.squeeze()
        ddm_surface_averaged = ddm_surface_averaged.squeeze()
        ctm_surface_averaged = ctm_surface_averaged.squeeze()
    # average over all data
    if sat_averaged_vcd.ndim == 4:
        sat_averaged_vcd = np.nanmean(np.nanmean(
            sat_averaged_vcd, axis=3).squeeze(), axis=2).squeeze()
        ctm_averaged_vcd = np.nanmean(np.nanmean(
            ctm_averaged_vcd, axis=3).squeeze(), axis=2).squeeze()
        sat_averaged_error = np.sqrt(np.nanmean(np.nanmean(
            sat_averaged_error**2, axis=3).squeeze(), axis=2).squeeze())
        sat_aux1 = np.nanmean(np.nanmean(
            sat_aux1, axis=3).squeeze(), axis=2).squeeze()
        sat_aux2 = np.nanmean(np.nanmean(
            sat_aux2, axis=3).squeeze(), axis=2).squeeze()
        if reader_obj.read_ddm == True:
            emis_averaged = np.nanmean(np.nanmean(
                emis_averaged, axis=3).squeeze(), axis=2).squeeze()
            ddm_averaged = np.nanmean(np.nanmean(
                ddm_averaged, axis=3).squeeze(), axis=2).squeeze()
            ddm_surface_averaged = np.nanmean(np.nanmean(
                ddm_surface_averaged, axis=3).squeeze(), axis=2).squeeze()
            emis_err_averaged = np.sqrt(np.nanmean(np.nanmean(
                emis_err_averaged**2, axis=3).squeeze(), axis=2).squeeze())
            ctm_surface_averaged = np.nanmean(np.nanmean(
                ctm_surface_averaged, axis=3).squeeze(), axis=2).squeeze()
    if sat_averaged_vcd.ndim == 3:
        sat_averaged_vcd = np.nanmean(sat_averaged_vcd, axis=2).squeeze()
        ctm_averaged_vcd = np.nanmean(ctm_averaged_vcd, axis=2).squeeze()
        # TODO: we should update this but we never average over several months or years
        sat_averaged_error = np.sqrt(np.nanmean(
            sat_averaged_error**2, axis=2).squeeze())
        sat_aux1 = np.nanmean(sat_aux1, axis=2).squeeze()
        sat_aux2 = np.nanmean(sat_aux2, axis=2).squeeze()
        if reader_obj.read_ddm == True:
            ddm_averaged = np.nanmean(ddm_averaged, axis=2).squeeze()
            ddm_surface_averaged = np.nanmean(
                ddm_surface_averaged, axis=2).squeeze()
            ctm_surface_averaged = np.nanmean(
                ctm_surface_averaged, axis=2).squeeze()
            emis_averaged = np.nanmean(emis_averaged, axis=2).squeeze()
            emis_err_averaged = np.sqrt(np.nanmean(
                emis_err_averaged**2, axis=2).squeeze())
    if reader_obj.read_ddm == False:
        ddm_averaged = []
        ddm_surface_averaged = []
        emis_averaged = []
        emis_err_averaged = []
        ctm_surface_averaged = []

    output = averaged_field(sat_averaged_vcd, sat_averaged_error, ctm_averaged_vcd, ctm_surface_averaged,
                            sat_aux1, sat_aux2, ddm_averaged, ddm_surface_averaged, emis_averaged, emis_err_averaged,
                            gap_chosen, time_chosen)
    return output
