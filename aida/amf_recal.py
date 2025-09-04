import numpy as np
from scipy import interpolate
from aida.interpolator import _upscaler
from scipy.spatial import Delaunay

def _flatten_time(time_obj):
    """Convert datetime object to float representation."""
    return (
        time_obj.year * 10000 +
        time_obj.month * 100 +
        time_obj.day +
        time_obj.hour / 24.0 +
        time_obj.minute / 60.0 / 24.0 +
        time_obj.second / 3600.0 / 24.0
    )

def _hour_only_time(time_obj):
    """Convert datetime object to hour fraction of the day."""
    return (
        time_obj.hour / 24.0 +
        time_obj.minute / 60.0 / 24.0 +
        time_obj.second / 3600.0 / 24.0
    )

def _time_lister(data, tagname='time'):
    time_data, time_data_hour_only, time_data_datetype = [], [], []
    for granule in data:
        times = getattr(granule, tagname)
        time_data.extend([_flatten_time(t) for t in times])
        time_data_hour_only.extend([_hour_only_time(t) for t in times])
        time_data_datetype.append(times)
    return np.array(time_data), np.array(time_data_hour_only), time_data_datetype

def _find_closest(time_sat, time_sat_hour_only, time_ctm, time_ctm_hour_only, averaged, freq=24):
    if not averaged:
        idx = np.argmin(np.abs(time_sat - time_ctm))
        day, hour = divmod(idx, freq+1)
    else:
        idx = np.argmin(np.abs(time_sat_hour_only - time_ctm_hour_only))
        day, hour = 0, idx
    return day, hour, idx

def _upscale_fields(field_3d, sat_coord, tri, ctm_lon, ctm_lat, gridsize_ctm, threshold_sat):
    shape = (field_3d.shape[0], *sat_coord["Longitude"].shape)
    upscaled = np.full(shape, np.nan)
    for z in range(field_3d.shape[0]):
        _, _, upscaled[z], _ = _upscaler(ctm_lon, ctm_lat, field_3d[z], sat_coord, gridsize_ctm, threshold_sat, tri=tri)
    return upscaled

def amf_recal(ctm_data, sat_data, ddm_data, ddm_read=False):

    print('AMF Recal begins...')
    time_ctm, time_ctm_hour_only, time_ctm_datetype = _time_lister(ctm_data)
    averaged = ctm_data[0].averaged

    if ddm_read:
        print('Synching DDM and SAT...')
        time_ddm, time_ddm_hour_only, time_ddm_datetype = _time_lister(ddm_data, 'time_ddm')
        time_emis, time_emis_hour_only, time_emis_datetype = _time_lister(ddm_data, 'time_emis')

    ctm_lon, ctm_lat = ctm_data[0].longitude, ctm_data[0].latitude
    points = np.column_stack((ctm_lon.flatten(), ctm_lat.flatten()))
    tri = Delaunay(points)

    for idx, L2_granule in enumerate(sat_data):
        if L2_granule is None: continue
        time_sat = _flatten_time(L2_granule.time)
        time_sat_hour_only = _hour_only_time(L2_granule.time)

        # Find closest indices
        day_ctm, hour_ctm, idx_ctm = _find_closest(time_sat, time_sat_hour_only, time_ctm, time_ctm_hour_only, averaged, freq=24 if averaged else 25)
        print(f"CMAQ file for L2 at {L2_granule.time}: {time_ctm_datetype[day_ctm][hour_ctm]}")
        if ddm_read:
            day_ddm, hour_ddm, _ = _find_closest(time_sat, time_sat_hour_only, time_ddm, time_ddm_hour_only, averaged)
            day_emis, hour_emis, _ = _find_closest(time_sat, time_sat_hour_only, time_emis, time_emis_hour_only, averaged)
            print(f"DDM file: {time_ddm_datetype[day_ddm][hour_ddm]}")
            print(f"EMIS file: {time_emis_datetype[day_emis][hour_emis]}")

        # Extract fields
        Mair, g, N_A = 28.97e-3, 9.80665, 6.02214076e23
        ctm = ctm_data[day_ctm]
        ctm_mid_pressure = np.squeeze(ctm.pressure_mid[hour_ctm])
        ctm_profile = np.squeeze(ctm.gas_profile[hour_ctm])
        ctm_deltap = np.squeeze(ctm.delta_p[hour_ctm])
        ctm_partial_column = ctm_deltap * ctm_profile / g / Mair * N_A * 1e-4 * 1e-15 * 100.0 * 1e-9

        if ddm_read:
            ddm = ddm_data[day_ddm]
            emis = ddm_data[day_emis]
            ddm_out = np.squeeze(ddm.ddm_out[hour_ddm])
            emis_total = np.squeeze(emis.emis_tot[hour_emis])
            emis_error = np.squeeze(emis.emis_err[hour_emis])
            dual = emis_total.ndim == 3

            if dual:
                ddm_partial = [ctm_deltap * ddm_out[..., k].squeeze() / g / Mair * N_A * 1e-4 * 1e-15 * 100.0 * 1e-9 for k in range(2)]
                emis_total = [emis_total[k,...] for k in range(2)]
                emis_error = [emis_error[k,...] for k in range(2)]
                ddm_surface = [ddm_out[0,:,:,k].squeeze() for k in range(2)]
            else:
                ddm_partial = [ctm_deltap * ddm_out / g / Mair * N_A * 1e-4 * 1e-15 * 100.0 * 1e-9]
                emis_total = [emis_total]
                emis_error = [emis_error]
                ddm_surface = [ddm_out[0,...].squeeze()]
            surface_conc = ctm_profile[0,...].squeeze()

        # Upscaling if needed
        if L2_granule.ctm_upscaled_needed == True:
            sat_coord = {"Longitude": L2_granule.longitude_center, "Latitude": L2_granule.latitude_center}
            size_grid_sat_lon = np.abs(sat_coord["Longitude"][0, 0]-sat_coord["Longitude"][0, 1])
            size_grid_sat_lat = np.abs(sat_coord["Latitude"][0, 0]-sat_coord["Latitude"][1, 0])
            threshold_sat = np.sqrt(size_grid_sat_lon**2 + size_grid_sat_lat**2)
            size_grid_model_lon = np.abs(ctm_lon[0, 0]-ctm_lon[0, 1])
            size_grid_model_lat = np.abs(ctm_lat[0, 0]-ctm_lat[1, 0])
            gridsize_ctm = np.sqrt(size_grid_model_lon**2 + size_grid_model_lat**2)

            ctm_mid_pressure = _upscale_fields(ctm_mid_pressure, sat_coord, tri, ctm_lon, ctm_lat, gridsize_ctm, threshold_sat)
            ctm_partial_column = _upscale_fields(ctm_deltap*ctm_profile/g/Mair*N_A*1e-4*1e-15*100.0*1e-9, sat_coord, tri, ctm_lon, ctm_lat, gridsize_ctm, threshold_sat)

            if ddm_read:
                dual_index = 2 if dual else 1
                ddm_partial = [ _upscale_fields(ddm_partial[d], sat_coord, tri, ctm_lon, ctm_lat, gridsize_ctm, threshold_sat) for d in range(dual_index) ]
                emis_total = [ _upscaler(ctm_lon, ctm_lat, emis_total[d], sat_coord, gridsize_ctm, threshold_sat, tri=tri)[2] for d in range(dual_index) ]
                emis_error = [ np.sqrt(_upscaler(ctm_lon, ctm_lat, emis_error[d]**2, sat_coord, gridsize_ctm, threshold_sat, tri=tri, error=True)[2]) for d in range(dual_index) ]
                ddm_surface = [ _upscaler(ctm_lon, ctm_lat, ddm_surface[d], sat_coord, gridsize_ctm, threshold_sat, tri=tri)[2] for d in range(dual_index) ]
                surface_conc = _upscaler(ctm_lon, ctm_lat, surface_conc, sat_coord, gridsize_ctm, threshold_sat, tri=tri)[2]

        # AMF recal or just VCD
        if np.size(L2_granule.scattering_weights) == 1:
            if ddm_read:
                raise Exception("Cannot do inversion without SWs.")
            print('No scattering weights found, recalculation is not possible..just grabbing VCDs')
            if np.size(L2_granule.tropopause) != 1:
                for z in range(ctm_profile.shape[0]):
                    mask = ctm_mid_pressure[z] < L2_granule.tropopause
                    ctm_partial_column[z][mask] = np.nan
            model_VCD = np.nansum(ctm_partial_column, axis=0)
            model_VCD[np.isnan(L2_granule.vcd)] = np.nan
            sat_data[idx].ctm_vcd = model_VCD
            sat_data[idx].ctm_time_at_sat = time_ctm[idx_ctm]
            sat_data[idx].old_amf = np.empty((1))
            sat_data[idx].new_amf = np.empty((1))
            continue

        # AMF recalculation loop
        vcd_shape = L2_granule.vcd.shape
        mask = np.isnan(L2_granule.vcd) | np.isinf(L2_granule.vcd)
        new_amf = np.full(vcd_shape, np.nan)
        model_VCD = np.full(vcd_shape, np.nan)
        ddm_vcd = np.full(vcd_shape, np.nan)
        ddm_vcd2 = np.full(vcd_shape, np.nan) if ddm_read and dual else None

        for i in range(vcd_shape[0]):
            for j in range(vcd_shape[1]):
                if mask[i, j]: continue
                ctm_partial_column_tmp = ctm_partial_column[:, i, j].squeeze()
                ctm_mid_pressure_tmp = ctm_mid_pressure[:, i, j].squeeze()
                # DDM
                if ddm_read:
                    if dual:
                        ctm_partial_ddm_0 = ddm_partial[0][:, i, j].squeeze()
                        ctm_partial_ddm_1 = ddm_partial[1][:, i, j].squeeze()
                    else:
                        ctm_partial_ddm_0 = ddm_partial[0][:, i, j].squeeze()
                # Interpolate scattering weights
                f = interpolate.interp1d(np.log(L2_granule.pressure_mid[:, i, j].squeeze()),
                                         L2_granule.scattering_weights[:, i, j].squeeze(), fill_value="extrapolate")
                interpolated_SW = f(np.log(ctm_mid_pressure_tmp))
                interpolated_SW[np.isinf(interpolated_SW)] = 0.0
                # Tropopause mask
                if np.size(L2_granule.tropopause) != 1:
                    tropo = L2_granule.tropopause[i, j]
                    tropo_mask = ctm_mid_pressure_tmp <= tropo
                    interpolated_SW[tropo_mask] = np.nan
                    ctm_partial_column_tmp[tropo_mask] = np.nan
                    if ddm_read:
                        ctm_partial_ddm_0[tropo_mask] = np.nan
                        if dual: ctm_partial_ddm_1[tropo_mask] = np.nan
                # Calculate SCD, VCD, AMF
                model_SCD = np.nansum(interpolated_SW * ctm_partial_column_tmp)
                model_VCD[i, j] = np.nansum(ctm_partial_column_tmp)
                new_amf[i, j] = model_SCD / model_VCD[i, j] if model_VCD[i, j] != 0 else np.nan
                # DDM VCD
                if ddm_read:
                    ddm_vcd[i, j] = np.nansum(ctm_partial_ddm_0)
                    if dual: ddm_vcd2[i, j] = np.nansum(ctm_partial_ddm_1)

        # Update sat_data
        sat_data[idx].old_amf = getattr(sat_data[idx], "scd", np.empty((1))) / getattr(sat_data[idx], "vcd", np.empty((1)))
        new_amf[mask] = np.nan
        sat_data[idx].new_amf = new_amf
        sat_data[idx].vcd = getattr(sat_data[idx], "scd", np.empty((1))) / new_amf
        model_VCD[mask] = np.nan
        sat_data[idx].ctm_vcd = model_VCD
        sat_data[idx].ctm_time_at_sat = time_ctm[idx_ctm]

        # DDM population
        if ddm_read:
            if dual:
                ddm_vcd2[mask] = np.nan
                for arr in [emis_total, emis_error, ddm_surface]:
                    arr = np.array(arr)
                    arr[mask,:] = np.nan
            else:
                for arr in [emis_total, emis_error, ddm_surface]:
                    arr = np.array(arr)
                    arr[mask] = np.nan
            ddm_vcd[mask] = np.nan
            surface_conc[mask] = np.nan
            sat_data[idx].ddm_vcd = np.stack((ddm_vcd, ddm_vcd2), axis=2)
            sat_data[idx].ddm_surface = ddm_surface
            sat_data[idx].ctm_surface_conc = surface_conc
            sat_data[idx].emis_tot = emis_total
            sat_data[idx].emis_err = emis_error

    return sat_data