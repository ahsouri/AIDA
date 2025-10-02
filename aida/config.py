import numpy as np
from dataclasses import dataclass
import datetime


@dataclass
class satellite_amf:
    vcd: np.ndarray
    amf: np.ndarray
    time: datetime.datetime
    tropopause: np.ndarray
    latitude_center: np.ndarray
    longitude_center: np.ndarray
    latitude_corner: np.ndarray
    longitude_corner: np.ndarray
    uncertainty: np.ndarray
    quality_flag: np.ndarray
    pressure_mid: np.ndarray
    scattering_weights: np.ndarray
    ctm_upscaled_needed: bool
    ctm_vcd: np.ndarray
    ctm_surface_conc: np.ndarray
    ctm_time_at_sat: datetime.datetime
    old_amf: np.ndarray
    new_amf: np.ndarray
    ddm_vcd: np.ndarray
    ddm_surface: np.ndarray
    emis_tot: np.ndarray
    emis_err: np.ndarray


@dataclass
class satellite_opt:
    vcd: np.ndarray
    time: datetime.datetime
    profile: np.ndarray
    tropopause: np.ndarray
    latitude_center: np.ndarray
    longitude_center: np.ndarray
    latitude_corner: np.ndarray
    longitude_corner: np.ndarray
    uncertainty: np.ndarray
    quality_flag: np.ndarray
    pressure_mid: np.ndarray
    averaging_kernels: np.ndarray
    ctm_upscaled_needed: bool
    ctm_vcd: np.ndarray
    ctm_xcol: np.ndarray
    ctm_time_at_sat: datetime.datetime
    aprior_column: np.ndarray
    apriori_profile: np.ndarray
    surface_pressure: np.ndarray
    apriori_surface: np.ndarray
    x_col: np.ndarray


@dataclass
class ctm_model:
    latitude: np.ndarray
    longitude: np.ndarray
    time: list
    gas_profile: np.ndarray
    pressure_mid: np.ndarray
    tempeature_mid: np.ndarray
    delta_p: np.ndarray
    ctmtype: str
    averaging: bool


@dataclass
class ddm_emis_model:
    time_ddm: list
    time_emis: list
    ddm_out: np.ndarray
    emis_tot: np.ndarray
    emis_err: np.ndarray
    averaging: bool


@dataclass
class averaged_field:
    sat_vcd: np.ndarray
    sat_err: np.ndarray
    ctm_vcd: np.ndarray
    ctm_surface: np.ndarray
    aux1: np.ndarray
    aux2: np.ndarray
    ddm_vcd: np.ndarray
    ddm_surface: np.ndarray
    emis_total: np.ndarray
    emis_error: np.ndarray
    gap_field: np.ndarray
    time_sat: np.ndarray


@dataclass
class OI_result:
    ctm_corrected: np.ndarray
    ak: np.ndarray
    increment: np.ndarray
    error_analysis: np.ndarray


@dataclass
class inversion_result:
    post_emis: np.array
    ak: np.array
    increment: np.array
    error_analysis: np.array
    ratio: np.array
