import yaml
from aida import aida
from pathlib import Path
import sys

# Read the control file
with open('./control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

ctm_name = ctrl_opts['ctm_name']
ctm_conc_dir = ctrl_opts['ctm_conc_dir']
ctm_mcip_dir = ctrl_opts['ctm_mcip_dir']
ctm_ddm_dir = ctrl_opts['ctm_ddm_dir']
ctm_emis_dir = ctrl_opts['ctm_emis_dir']
ctm_avg = False
state_vectors = ctrl_opts['state_vectors']
state_vector_conc_errors = ctrl_opts['state_vector_conc_errors']
state_vector_nox_errors = ctrl_opts['state_vector_nox_errors']
state_vector_voc_errors = ctrl_opts['state_vector_voc_errors']
sensor = ctrl_opts['sensor']
troposphere_no2_only = ctrl_opts['troposphere_no2_only']
sat_path = ctrl_opts['sat_path']
output_pdf_dir = ctrl_opts['output_pdf_dir']
output_nc_dir = ctrl_opts['output_nc_dir']
num_job = ctrl_opts['num_job']
sensor = ctrl_opts['sensor']
validation_only = ctrl_opts['validation']
save_daily = ctrl_opts['save_daily']
bool_iteration = ctrl_opts['first_estimate']
inversion_previous_folder = ctrl_opts['inversion_prev']
bias_sat = ctrl_opts['bias_sat']


year = int(sys.argv[1])
month = int(sys.argv[2])

# looping over state vectors (each is done separately)
cnt = -1
for statev in state_vectors:
    cnt = cnt + 1
    # doing nox inversion
    if statev == 'NOx':
        do_inversion = True
        do_oi = False
        read_ddm = True
        gasname = 'NO2'
        state_err = state_vector_nox_errors
    # doing voc inversion
    elif statev == 'VOC':
        do_inversion = True
        do_oi = False
        read_ddm = True
        gasname = 'HCHO'
        state_err = state_vector_voc_errors
    # doing no2, hcho, or co assimilation
    else:
        do_inversion = False
        do_oi = True
        read_ddm = False
        gasname = statev
        if statev == 'NO2':
            state_err = state_vector_conc_errors[0]
        if statev == 'HCHO':
            state_err = state_vector_conc_errors[1]
        if statev == 'CO':
            state_err = state_vector_conc_errors[2]
    # do only validation (NOx or VOC should be dropped from state_vectors)
    if validation_only == True:
        if (statev == 'NOx' or statev == 'VOC'):
            continue
        do_inversion = False
        do_oi = False
        read_ddm = False
    # calling AIDA
    aida_obj = aida()
    aida_obj.read_data(ctm_name, Path(ctm_conc_dir), Path(ctm_mcip_dir), gasname, sensor[cnt] + '_' + gasname, Path(sat_path[cnt]), str(year) + f"{month:02}",
                       (state_err), Path(ctm_ddm_dir[cnt]), Path(ctm_emis_dir), read_ddm=read_ddm, averaged=ctm_avg, read_ak=True, trop=troposphere_no2_only, num_job=num_job)

    if sensor[cnt] == "MOPITT":
        aida_obj.conv_ak()
    else:
        aida_obj.recal_amf()

    if save_daily:
        aida_obj.savedaily(output_nc_dir, gasname,
                           str(year) + '_' + f"{month:02}")

    if month != 12:
        aida_obj.average(str(
            year) + '-' + f"{month:02}" + '-01', str(year) + '-' + f"{month+1:02}" + '-01')
    else:
        aida_obj.average(
            str(year) + '-' + f"{month:02}" + '-01', str(year+1) + '-' + "01" + '-01')

    if bias_sat == True:
        aida_obj.bias_correct(sensor[cnt], gasname)

    if do_oi == True:
        aida_obj.oi(error_ctm=state_err)

    if bool_iteration == False:
        aida_obj.emission_last_iter(
            inversion_previous_folder, str(year) + f"{month:02}")

    if do_inversion == True:
        aida_obj.inversion(
            gasname, sensor[cnt])

    aida_obj.reporting(gasname + '_' + str(year) +
                       f"{month:02}" + '_', gasname, output_pdf_dir)
    aida_obj.write_to_nc(gasname + '_' + str(year) +
                         f"{month:02}" + '_', output_nc_dir, read_ddm=read_ddm)
