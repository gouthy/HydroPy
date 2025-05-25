"""Process-based distributed hydrologic model using JAX."""


from __future__ import annotations

from typing import NamedTuple

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


def _require_jax():  # pragma: no cover - helper for missing dependency
    if jnp is None:
        raise ImportError(
            "JAX is required to use hydropy.model. Please install jax."
        )


class SnowParams(NamedTuple):
    """Parameters for :func:`snow_module`."""

    T_sn_min: float = 272.05
    T_sn_max: float = 276.45
    T_melt: float = 273.15
    f_snlq_max: float = 0.06


class CanopyParams(NamedTuple):
    """Parameters for :func:`skin_canopy_module`."""

    f_bare: float
    f_veg: float
    LAI: float
    cap0: float


class SoilParams(NamedTuple):
    """Parameters for :func:`soil_module`."""

    S_so_max: float
    S_so_wilt: float
    S_so_grmin: float
    S_so_grmax: float
    S_so_sg_min: float
    S_so_sg_max: float
    b: float
    R_gr_min: float
    R_gr_max: float
    dt: float
    f_so_crit: float = 0.75
    f_so_bs_low: float = 0.05
    R_gr_exp: float = 1.5


class GroundwaterParams(NamedTuple):
    """Parameters for :func:`surface_and_gw_module`."""

    f_lake: float
    f_wetland: float
    LAG_sw: float
    LAG_gw: float


class HydroParams(NamedTuple):

    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


class HydroState(NamedTuple):
    """State variables for the hydrologic model."""

    S_sn: float
    S_snlq: float
    S_skin: float
    S_can: float
    S_so: float
    S_sw: float
    S_gw: float


def update_snow_state(
    state: HydroState, P: float, T_srf: float, f_day: float, params: HydroParams
) -> tuple[HydroState, dict]:
    """Update snow state using :func:`snow_module`."""

    snow = snow_module(
        P,
        T_srf,
        f_day,
        state.S_sn,
        state.S_snlq,
        params.snow.T_sn_min,
        params.snow.T_sn_max,
        params.snow.T_melt,
        params.snow.f_snlq_max,
    )
    state = state._replace(S_sn=snow["S_sn_new"], S_snlq=snow["S_snlq_new"])
    return state, snow


def update_canopy_state(
    state: HydroState,
    P: float,
    snow: dict,
    PET: float,
    params: HydroParams,
) -> tuple[HydroState, dict]:
    """Update skin and canopy storages."""

    canopy = skin_canopy_module(
        P,
        snow["P_sn"],
        snow["R_sn"],
        PET,
        params.canopy.f_bare,
        params.canopy.f_veg,
        state.S_skin,
        state.S_can,
        params.canopy.LAI,
        params.canopy.cap0,
    )
    state = state._replace(S_skin=canopy["S_skin_new"], S_can=canopy["S_can_new"])
    return state, canopy


def update_soil_state(
    state: HydroState,
    P: float,
    T_srf: float,
    PET: float,
    snow: dict,
    canopy: dict,
    params: HydroParams,
) -> tuple[HydroState, dict]:
    """Update soil moisture dynamics."""

    soil = soil_module(
        P,
        T_srf,
        PET,
        snow,
        canopy,
        params.canopy.f_bare,
        params.canopy.f_veg,
        state.S_so,
        params.soil.S_so_max,
        params.soil.S_so_wilt,
        params.soil.S_so_grmin,
        params.soil.S_so_grmax,
        params.soil.S_so_sg_min,
        params.soil.S_so_sg_max,
        params.soil.b,
        params.soil.R_gr_min,
        params.soil.R_gr_max,
        params.soil.dt,
        params.soil.f_so_crit,
        params.soil.f_so_bs_low,
        params.soil.R_gr_exp,
    )
    state = state._replace(S_so=soil["S_so_new"])
    return state, soil


def update_routing_state(
    state: HydroState,
    P: float,
    snow: dict,
    soil: dict,
    PET: float,
    params: HydroParams,
) -> tuple[HydroState, dict]:
    """Update surface water and groundwater states."""

    routing = surface_and_gw_module(
        P,
        snow,
        soil,
        PET,
        params.groundwater.f_lake,
        params.groundwater.f_wetland,
        state.S_sw,
        state.S_gw,
        params.groundwater.LAG_sw,
        params.groundwater.LAG_gw,
    )
    state = state._replace(S_sw=routing["S_sw_new"], S_gw=routing["S_gw_new"])
    return state, routing


def snow_module(
    P: float,
    T_srf: float,
    f_day: float,
    S_sn: float,
    S_snlq: float,
    T_sn_min: float = 272.05,
    T_sn_max: float = 276.45,
    T_melt: float = 273.15,
    f_snlq_max: float = 0.06,
):
    """Snow accumulation and melt."""

    frac_snow = (T_sn_max - T_srf) / (T_sn_max - T_sn_min)
    frac_snow = jnp.clip(frac_snow, 0.0, 1.0)
    P_sn = P * frac_snow

    DDF = f_day * 8.3 + 0.7
    melt_pot = jnp.maximum(0.0, DDF * (T_srf - T_melt))
    R_sn_raw = jnp.minimum(melt_pot, S_sn + P_sn)

    max_liquid = f_snlq_max * (S_sn + P_sn)
    possible_retention = max_liquid - S_snlq
    F_snlq = jnp.clip(R_sn_raw, 0.0, possible_retention)

    R_sn = R_sn_raw - F_snlq
    S_snlq_new = S_snlq + F_snlq

    refreeze = jnp.where(T_srf < T_melt, S_snlq_new, 0.0)
    S_snlq_new = jnp.where(T_srf < T_melt, 0.0, S_snlq_new)

    dS_sn = P_sn - R_sn + refreeze
    S_sn_new = S_sn + dS_sn

    return {
        "P_sn": P_sn,
        "R_sn": R_sn,
        "F_snlq": F_snlq,
        "refreeze": refreeze,
        "S_sn_new": S_sn_new,
        "S_snlq_new": S_snlq_new,
    }


def skin_canopy_module(
    P: float,
    P_sn: float,
    R_sn: float,
    PET: float,
    f_bare: float,
    f_veg: float,
    S_skin: float,
    S_can: float,
    LAI: float,
    cap0: float,
):
    """Compute interception and evaporation for skin and canopy."""

    P_ra = P - P_sn

    def _layer(f_lct, S_prc, S_max):
        S_norm = jnp.where(f_lct > 0.0, S_prc / f_lct, 0.0)
        avail = S_norm + P_ra + R_sn
        f_wet = jnp.where(S_max > 0.0, jnp.minimum(1.0, avail / S_max), 0.0)
        E_norm = jnp.minimum(avail, PET * f_wet)
        R_norm = jnp.maximum(0.0, avail - E_norm - S_max)
        S_norm_new = jnp.minimum(S_max, avail - E_norm)
        return (
            S_norm_new * f_lct,
            E_norm * f_lct,
            R_norm * f_lct,
        )

    S_skin_new, E_skin_tot, R_skin_tot = _layer(f_bare, S_skin, cap0)
    S_can_new, E_can_tot, R_can_tot = _layer(f_veg, S_can, cap0 * LAI)

    return {
        "S_skin_new": S_skin_new,
        "E_skin_tot": E_skin_tot,
        "R_skin_tot": R_skin_tot,
        "S_can_new": S_can_new,
        "E_can_tot": E_can_tot,
        "R_can_tot": R_can_tot,
        "R_tr": R_skin_tot + R_can_tot,
    }


def soil_module(
    P: float,
    T_srf: float,
    PET: float,
    snow: dict,
    canopy: dict,
    f_bare: float,
    f_veg: float,
    S_so: float,
    S_so_max: float,
    S_so_wilt: float,
    S_so_grmin: float,
    S_so_grmax: float,
    S_so_sg_min: float,
    S_so_sg_max: float,
    b: float,
    R_gr_min: float,
    R_gr_max: float,
    dt: float,
    f_so_crit: float = 0.75,
    f_so_bs_low: float = 0.05,
    R_gr_exp: float = 1.5,
):
    """Soil moisture processes."""

    P_sn = snow["P_sn"]
    R_sn = snow["R_sn"]
    R_tr = canopy["R_tr"]
    E_skin = canopy["E_skin_tot"]
    E_can = canopy["E_can_tot"]

    denom_T = f_so_crit * S_so_max - S_so_wilt
    theta_T = jnp.where(
        denom_T > 0,
        jnp.clip((S_so - S_so_wilt) / denom_T, 0.0, 1.0),
        0.0,
    )
    E_T = (PET - E_can) * theta_T * f_veg

    denom_bs = (1.0 - f_so_bs_low) * S_so_max
    theta_bs = jnp.where(
        denom_bs > 0,
        jnp.clip((S_so - f_so_bs_low * S_so_max) / denom_bs, 0.0, 1.0),
        0.0,
    )
    E_bs = (PET - E_skin) * theta_bs * f_bare

    def _sg_capacity(S_so):
        frac = (S_so - S_so_sg_min) / (S_so_max - S_so_sg_min)
        return jnp.where(
            S_so > S_so_sg_min,
            S_so_sg_max - (S_so_sg_max - S_so_sg_min) * (1.0 - frac) ** (1.0 / (1.0 + b)),
            S_so,
        )

    S_so_sg = _sg_capacity(S_so)

    delta_sg = S_so_sg_max - S_so_sg_min
    c1 = jnp.where(
        delta_sg > 0,
        jnp.minimum(1.0, ((S_so_sg_max - S_so_sg) / delta_sg) ** (1.0 + b)),
        1.0,
    )
    c2 = jnp.where(
        delta_sg > 0,
        jnp.maximum(0.0, ((S_so_sg_max - S_so_sg - R_tr) / delta_sg) ** (1.0 + b)),
        0.0,
    )

    R_srf = jnp.where(
        T_srf < 273.15,
        R_tr,
        jnp.where(
            (R_tr <= 0) | ((S_so_sg + R_tr) <= S_so_sg_min),
            0.0,
            jnp.where(
                (S_so_sg + R_tr) > S_so_sg_max,
                R_tr + jnp.maximum(0.0, S_so - S_so_max),
                R_tr - jnp.maximum(0.0, S_so_sg_min - S_so) - (delta_sg / (1.0 + b)) * (c1 - c2),
            ),
        ),
    )

    R_gr_low = jnp.where(S_so_max > 0, R_gr_min * dt * (S_so / S_so_max), 0.0)
    frac2 = jnp.where(
        S_so > S_so_grmax,
        (S_so - S_so_grmax) / jnp.maximum(S_so_max - S_so_grmax, 0.0),
        0.0,
    )
    R_gr_high = jnp.where(S_so > S_so_grmax, (R_gr_max - R_gr_min) * dt * (frac2 ** R_gr_exp), 0.0)

    R_gr = jnp.where(
        (S_so <= S_so_grmin) | (T_srf < 273.15),
        0.0,
        jnp.where(S_so <= S_so_grmax, R_gr_low, R_gr_low + R_gr_high),
    )

    dS_so = R_tr - R_srf - R_gr - E_T - E_bs
    S_so_new = S_so + dS_so

    return {
        "E_T": E_T,
        "E_bs": E_bs,
        "R_srf": R_srf,
        "R_gr": R_gr,
        "ΔS_so": dS_so,
        "S_so_new": S_so_new,
    }


def surface_and_gw_module(
    P: float,
    snow: dict,
    soil: dict,
    PET: float,
    f_lake: float,
    f_wetland: float,
    S_sw: float,
    S_gw: float,
    LAG_sw: float,
    LAG_gw: float,
):
    """Surface-water and groundwater balance."""

    P_sn = snow["P_sn"]
    R_sn = snow["R_sn"]
    R_srf = soil["R_srf"]
    R_gr = soil["R_gr"]

    P_ra = P - P_sn

    f_sw = jnp.maximum(f_lake, f_wetland)
    R_sw = S_sw / (LAG_sw + 1.0)
    dS_sw = (P_ra + R_sn - PET) * f_sw + R_srf - R_sw
    S_sw_new = S_sw + dS_sw

    R_gw = S_gw / (LAG_gw + 1.0)
    dS_gw = R_gr - R_gw
    S_gw_new = S_gw + dS_gw

    return {
        "f_sw": f_sw,
        "P_ra": P_ra,
        "R_sw": R_sw,
        "ΔS_sw": dS_sw,
        "S_sw_new": S_sw_new,
        "R_gw": R_gw,
        "ΔS_gw": dS_gw,
        "S_gw_new": S_gw_new,
    }


def _single_cell_model(
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,
) -> jnp.ndarray:
    _require_jax()

    def step(state: HydroState, inputs):
        p, et, t = inputs
        # day-length fraction is not provided -> assume 1.0
        state, snow = update_snow_state(state, p, t, 1.0, params)
        state, canopy = update_canopy_state(state, p, snow, et, params)
        state, soil = update_soil_state(state, p, t, et, snow, canopy, params)
        state, rout = update_routing_state(state, p, snow, soil, et, params)
        runoff = rout["R_sw"] + rout["R_gw"]
        return state, runoff

    init = HydroState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    _, runoff = jax.lax.scan(step, init, (precip, pet, temp))
    return runoff


def hydrologic_model(
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,
) -> jnp.ndarray:
    """Run distributed hydrologic model.

    Parameters are shared across all locations.
    Inputs are arrays of shape (time, n_locations).
    """
    _require_jax()
    return jax.vmap(_single_cell_model, in_axes=(1,1,1,None), out_axes=1)(precip, pet, temp, params)



__all__ = [
    "SnowParams",
    "CanopyParams",
    "SoilParams",
    "GroundwaterParams",
    "HydroParams",
    "hydrologic_model",
]
