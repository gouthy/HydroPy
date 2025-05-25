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
    day_frac: float
    T_sn_min: float = 272.05
    T_sn_max: float = 276.45
    T_melt: float = 273.15
    f_snlq_max: float = 0.06


class CanopyParams(NamedTuple):

    f_bare: float
    f_veg: float
    LAI: float
    cap0: float


class SoilParams(NamedTuple):
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
    f_lake: float = 0.0
    f_wetland: float = 0.0
    LAG_sw: float = 0.0
    LAG_gw: float = 0.0


class HydroParams(NamedTuple):
    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


class HydroState(NamedTuple):
    S_sn: float
    S_snlq: float
    S_skin: float
    S_can: float
    S_so: float
    S_sw: float
    groundwater: float


# --- process modules -------------------------------------------------------

def snow_module(P: float, T_srf: float, params: SnowParams, S_sn: float, S_snlq: float):
    """Snow accumulation and melt module."""
    frac_snow = (params.T_sn_max - T_srf) / (params.T_sn_max - params.T_sn_min)
    frac_snow = jnp.clip(frac_snow, 0.0, 1.0)
    P_sn = P * frac_snow

    DDF = params.day_frac * 8.3 + 0.7
    melt_pot = jnp.maximum(0.0, DDF * (T_srf - params.T_melt))
    R_sn_raw = jnp.minimum(melt_pot, S_sn + P_sn)

    max_liquid = params.f_snlq_max * (S_sn + P_sn)
    possible_retention = max_liquid - S_snlq
    F_snlq = jnp.clip(R_sn_raw, 0.0, possible_retention)
    R_sn = R_sn_raw - F_snlq

    S_snlq_new = S_snlq + F_snlq
    refreeze = jnp.where(T_srf < params.T_melt, S_snlq_new, 0.0)
    S_snlq_new = jnp.where(T_srf < params.T_melt, 0.0, S_snlq_new)

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
    params: CanopyParams,
    S_skin: float,
    S_can: float,
):
    """Skin and canopy interception and evaporation."""
    P_ra = P - P_sn
    out = {}
    for prc, f_lct, S_prc in [
        ("skin", params.f_bare, S_skin),
        ("can", params.f_veg, S_can),
    ]:
        def handle():
            S_norm = S_prc / f_lct
            S_max = params.cap0 * (1.0 if prc == "skin" else params.LAI)
            avail = S_norm + P_ra + R_sn
            f_wet = jnp.where(S_max > 0.0, jnp.minimum(1.0, avail / S_max), 0.0)
            E_norm = jnp.minimum(avail, PET * f_wet)
            R_norm = jnp.maximum(0.0, avail - E_norm - S_max)
            S_norm_new = jnp.minimum(S_max, avail - E_norm)
            return {
                f"S_{prc}_new": S_norm_new * f_lct,
                f"E_{prc}_tot": E_norm * f_lct,
                f"R_{prc}_tot": R_norm * f_lct,
            }

        res = jax.lax.cond(
            f_lct <= 0.0,
            lambda: {f"S_{prc}_new": 0.0, f"E_{prc}_tot": 0.0, f"R_{prc}_tot": 0.0},
            handle,
        )
        out.update(res)

    out["R_tr"] = out["R_skin_tot"] + out["R_can_tot"]
    return out


def soil_module(
    P: float,
    T_srf: float,
    PET: float,
    snow: dict,
    canopy: dict,
    params: SoilParams,
    S_so: float,
    canopy_params: CanopyParams,
):
    """Soil moisture processes."""
    P_sn = snow["P_sn"]
    R_sn = snow["R_sn"]
    R_tr = canopy["R_tr"]
    E_skin = canopy["E_skin_tot"]
    E_can = canopy["E_can_tot"]
    f_bare = canopy_params.f_bare
    f_veg = canopy_params.f_veg

    denom_T = params.f_so_crit * params.S_so_max - params.S_so_wilt
    theta_T = jnp.where(
        denom_T > 0,
        jnp.clip((S_so - params.S_so_wilt) / denom_T, 0.0, 1.0),
        0.0,
    )
    E_T = (PET - E_can) * theta_T * f_veg

    denom_bs = (1.0 - params.f_so_bs_low) * params.S_so_max
    theta_bs = jnp.where(
        denom_bs > 0,
        jnp.clip((S_so - params.f_so_bs_low * params.S_so_max) / denom_bs, 0.0, 1.0),
        0.0,
    )
    E_bs = (PET - E_skin) * theta_bs * f_bare

    S_so_sg = jnp.where(
        S_so > params.S_so_sg_min,
        params.S_so_sg_max
        - (params.S_so_sg_max - params.S_so_sg_min)
        * (1.0 - (S_so - params.S_so_sg_min) / (params.S_so_max - params.S_so_sg_min))
        ** (1.0 / (1.0 + params.b)),
        S_so,
    )


    delta_sg = params.S_so_sg_max - params.S_so_sg_min
    c1 = jnp.where(
        delta_sg > 0,
        jnp.minimum(1.0, ((params.S_so_sg_max - S_so_sg) / delta_sg) ** (1.0 + params.b)),
        1.0,
    )
    c2 = jnp.where(
        delta_sg > 0,
        jnp.maximum(
            0.0,
            ((params.S_so_sg_max - S_so_sg - R_tr) / delta_sg) ** (1.0 + params.b),
        ),
        0.0,
    )

    cond1 = T_srf < 273.15
    cond2 = jnp.logical_or(R_tr <= 0, (S_so_sg + R_tr) <= params.S_so_sg_min)
    cond3 = (S_so_sg + R_tr) > params.S_so_sg_max
    R_srf = jax.lax.select(
        cond1,
        R_tr,
        jax.lax.select(
            cond2,
            0.0,
            jax.lax.select(
                cond3,
                R_tr + jnp.maximum(0.0, S_so - params.S_so_max),
                R_tr
                - jnp.maximum(0.0, params.S_so_sg_min - S_so)
                - (delta_sg / (1.0 + params.b)) * (c1 - c2),
            ),
        ),
    )

    R_gr_low = jnp.where(
        params.S_so_max > 0,
        params.R_gr_min * params.dt * (S_so / params.S_so_max),
        0.0,
    )
    frac2 = jnp.where(
        (params.S_so_max - params.S_so_grmax) > 0,
        (S_so - params.S_so_grmax) / (params.S_so_max - params.S_so_grmax),
        0.0,
    )
    R_gr_high = jnp.where(
        S_so > params.S_so_grmax,
        (params.R_gr_max - params.R_gr_min) * params.dt * (frac2 ** params.R_gr_exp),
        0.0,
    )

    cond_low = jnp.logical_or(S_so <= params.S_so_grmin, T_srf < 273.15)
    cond_mid = S_so <= params.S_so_grmax
    R_gr = jax.lax.select(
        cond_low,
        0.0,
        jax.lax.select(cond_mid, R_gr_low, R_gr_low + R_gr_high),
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


def groundwater_process(
    P: float,
    snow: dict,
    soil: dict,
    PET: float,
    params: GroundwaterParams,
    S_sw: float,
    S_gw: float,
):
    """Surface water and shallow groundwater processes."""
    P_sn = snow["P_sn"]
    R_sn = snow["R_sn"]
    R_srf = soil["R_srf"]
    R_gr = soil["R_gr"]

    P_ra = P - P_sn
    f_sw = jnp.maximum(params.f_lake, params.f_wetland)

    R_sw = S_sw / (params.LAG_sw + 1.0)
    dS_sw = (P_ra + R_sn - PET) * f_sw + R_srf - R_sw
    S_sw_new = S_sw + dS_sw

    R_gw = S_gw / (params.LAG_gw + 1.0)
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


# --- state update wrappers -------------------------------------------------

def update_snow_state(state: HydroState, P: float, T: float, params: HydroParams):
    snow = snow_module(P, T, params.snow, state.S_sn, state.S_snlq)
    state = state._replace(S_sn=snow["S_sn_new"], S_snlq=snow["S_snlq_new"])
    return state, snow


def update_canopy_state(
    state: HydroState,
    P: float,
    snow: dict,
    PET: float,
    params: HydroParams,
):
    canopy = skin_canopy_module(
        P,
        snow["P_sn"],
        snow["R_sn"],
        PET,
        params.canopy,
        state.S_skin,
        state.S_can,
    )
    state = state._replace(S_skin=canopy["S_skin_new"], S_can=canopy["S_can_new"])
    return state, canopy


def update_soil_state(
    state: HydroState,
    P: float,
    T: float,
    PET: float,
    snow: dict,
    canopy: dict,
    params: HydroParams,
):
    soil = soil_module(
        P,
        T,
        PET,
        snow,
        canopy,
        params.soil,
        state.S_so,
        params.canopy,
    )
    state = state._replace(S_so=soil["S_so_new"])
    return state, soil


def update_groundwater_state(
    state: HydroState,
    P: float,
    PET: float,
    snow: dict,
    soil: dict,
    params: HydroParams,
) -> tuple[HydroState, float, float]:
    res = groundwater_process(
        P,
        snow,
        soil,
        PET,
        params.groundwater,
        state.S_sw,
        state.groundwater,
    )
    state = state._replace(S_sw=res["S_sw_new"], groundwater=res["S_gw_new"])
    return state, res["R_sw"], res["R_gw"]


# --- model ----------------------------------------------------------------

def _single_cell_model(
    precip: jnp.ndarray,
    evap: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,

) -> jnp.ndarray:
    _require_jax()

    def step(state: HydroState, inputs):
        p, e, t = inputs
        state, snow = update_snow_state(state, p, t, params)
        state, canopy = update_canopy_state(state, p, snow, e, params)
        state, soil = update_soil_state(state, p, t, e, snow, canopy, params)
        state, R_sw, R_gw = update_groundwater_state(state, p, e, snow, soil, params)
        runoff = R_sw + R_gw
        return state, runoff

    init = HydroState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    _, runoff = jax.lax.scan(step, init, (precip, evap, temp))
    return runoff


def hydrologic_model(
    precip: jnp.ndarray,
    evap: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,

) -> jnp.ndarray:
    """Run distributed hydrologic model.

    Parameters are shared across all locations.
    Inputs are arrays of shape (time, n_locations).
    """
    _require_jax()
    return jax.vmap(_single_cell_model, in_axes=(1, 1, 1, None), out_axes=1)(
        precip, evap, temp, params
    )

__all__ = [
    "SnowParams",
    "CanopyParams",
    "SoilParams",
    "GroundwaterParams",
    "HydroParams",
    "hydrologic_model",
]
