"""Process-based distributed hydrologic model using JAX."""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple, Dict, Any

try:
    import jax
    import jax.numpy as jnp
    from jax.tree_util import tree_map
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None
    tree_map = None


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


class LandUseParams(NamedTuple):
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams
    imperv_frac: float
    crop_coeff: float = 1.0


class HydroParams(NamedTuple):
    snow: SnowParams
    landuse: LandUseParams


class HydroState(NamedTuple):
    S_sn: float
    S_snlq: float
    S_skin: float
    S_can: float
    S_so: float
    S_sw: float
    groundwater: float


# --- land-use lookup table -----------------------------------------------

def default_landuse_lookup() -> Dict[str, LandUseParams]:
    """Return default land-use parameters."""
    return {
        "ENF": LandUseParams(
            canopy=CanopyParams(f_bare=0.0, f_veg=1.0, LAI=5.5, cap0=0.25),
            soil=SoilParams(
                S_so_max=250.0,
                S_so_wilt=60.0,
                S_so_grmin=30.0,
                S_so_grmax=180.0,
                S_so_sg_min=10.0,
                S_so_sg_max=200.0,
                b=0.35,
                R_gr_min=1e-4,
                R_gr_max=1e-3,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=3.0, LAG_gw=12.0),
            imperv_frac=0.0,
            crop_coeff=1.0,
        ),
        "DBF": LandUseParams(
            canopy=CanopyParams(f_bare=0.0, f_veg=1.0, LAI=6.0, cap0=0.30),
            soil=SoilParams(
                S_so_max=220.0,
                S_so_wilt=50.0,
                S_so_grmin=25.0,
                S_so_grmax=160.0,
                S_so_sg_min=10.0,
                S_so_sg_max=180.0,
                b=0.30,
                R_gr_min=1e-4,
                R_gr_max=2e-3,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=2.5, LAG_gw=10.0),
            imperv_frac=0.0,
            crop_coeff=1.0,
        ),
        "MF": LandUseParams(
            canopy=CanopyParams(f_bare=0.0, f_veg=1.0, LAI=5.8, cap0=0.28),
            soil=SoilParams(
                S_so_max=230.0,
                S_so_wilt=55.0,
                S_so_grmin=28.0,
                S_so_grmax=170.0,
                S_so_sg_min=10.0,
                S_so_sg_max=190.0,
                b=0.32,
                R_gr_min=1e-4,
                R_gr_max=1.5e-3,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=3.0, LAG_gw=11.0),
            imperv_frac=0.0,
            crop_coeff=1.0,
        ),
        "OS": LandUseParams(
            canopy=CanopyParams(f_bare=0.2, f_veg=0.8, LAI=1.5, cap0=0.10),
            soil=SoilParams(
                S_so_max=100.0,
                S_so_wilt=30.0,
                S_so_grmin=15.0,
                S_so_grmax=80.0,
                S_so_sg_min=5.0,
                S_so_sg_max=90.0,
                b=0.20,
                R_gr_min=5e-4,
                R_gr_max=2e-3,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=1.5, LAG_gw=7.0),
            imperv_frac=0.0,
            crop_coeff=1.0,
        ),
        "GRA": LandUseParams(
            canopy=CanopyParams(f_bare=0.3, f_veg=0.7, LAI=2.0, cap0=0.12),
            soil=SoilParams(
                S_so_max=120.0,
                S_so_wilt=35.0,
                S_so_grmin=18.0,
                S_so_grmax=90.0,
                S_so_sg_min=5.0,
                S_so_sg_max=100.0,
                b=0.22,
                R_gr_min=5e-4,
                R_gr_max=2.5e-3,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=1.0, LAG_gw=8.0),
            imperv_frac=0.0,
            crop_coeff=1.0,
        ),
        "SAV": LandUseParams(
            canopy=CanopyParams(f_bare=0.25, f_veg=0.75, LAI=3.5, cap0=0.15),
            soil=SoilParams(
                S_so_max=140.0,
                S_so_wilt=40.0,
                S_so_grmin=20.0,
                S_so_grmax=100.0,
                S_so_sg_min=8.0,
                S_so_sg_max=120.0,
                b=0.25,
                R_gr_min=5e-4,
                R_gr_max=3e-3,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=1.5, LAG_gw=9.0),
            imperv_frac=0.0,
            crop_coeff=1.0,
        ),
        "CROP": LandUseParams(
            canopy=CanopyParams(f_bare=0.2, f_veg=0.8, LAI=4.0, cap0=0.18),
            soil=SoilParams(
                S_so_max=160.0,
                S_so_wilt=45.0,
                S_so_grmin=22.0,
                S_so_grmax=110.0,
                S_so_sg_min=10.0,
                S_so_sg_max=130.0,
                b=0.25,
                R_gr_min=1e-3,
                R_gr_max=4e-3,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=2.0, LAG_gw=10.0),
            imperv_frac=0.0,
            crop_coeff=1.2,
        ),
        "URB": LandUseParams(
            canopy=CanopyParams(f_bare=0.1, f_veg=0.2, LAI=1.0, cap0=0.05),
            soil=SoilParams(
                S_so_max=60.0,
                S_so_wilt=15.0,
                S_so_grmin=8.0,
                S_so_grmax=35.0,
                S_so_sg_min=2.0,
                S_so_sg_max=50.0,
                b=0.1,
                R_gr_min=1e-5,
                R_gr_max=1e-4,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.0, LAG_sw=1.0, LAG_gw=5.0),
            imperv_frac=0.7,
            crop_coeff=1.0,
        ),
        "WET": LandUseParams(
            canopy=CanopyParams(f_bare=0.0, f_veg=1.0, LAI=5.0, cap0=0.20),
            soil=SoilParams(
                S_so_max=80.0,
                S_so_wilt=0.0,
                S_so_grmin=0.0,
                S_so_grmax=20.0,
                S_so_sg_min=0.0,
                S_so_sg_max=80.0,
                b=0.1,
                R_gr_min=0.0,
                R_gr_max=0.0,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=0.0, f_wetland=0.8, LAG_sw=4.0, LAG_gw=15.0),
            imperv_frac=0.0,
            crop_coeff=1.0,
        ),
        "WATR": LandUseParams(
            canopy=CanopyParams(f_bare=1.0, f_veg=0.0, LAI=0.0, cap0=0.0),
            soil=SoilParams(
                S_so_max=0.0,
                S_so_wilt=0.0,
                S_so_grmin=0.0,
                S_so_grmax=0.0,
                S_so_sg_min=0.0,
                S_so_sg_max=0.0,
                b=0.0,
                R_gr_min=0.0,
                R_gr_max=0.0,
                dt=86400.0,
            ),
            groundwater=GroundwaterParams(f_lake=1.0, f_wetland=0.0, LAG_sw=0.0, LAG_gw=0.0),
            imperv_frac=0.0,
            crop_coeff=0.0,
        ),
    }

# --- process modules -------------------------------------------------------

def snow_module(P: float, T_srf: float, params: SnowParams, S_sn: float, S_snlq: float):
    """Snow accumulation, melt, retention, and refreeze (mass-conserving)."""
    frac_snow = (params.T_sn_max - T_srf) / (params.T_sn_max - params.T_sn_min)
    frac_snow = jnp.clip(frac_snow, 0.0, 1.0)
    P_sn = P * frac_snow

    DDF = params.day_frac * 8.3 + 0.7
    melt_pot = jnp.maximum(0.0, DDF * (T_srf - params.T_melt))
    R_sn_raw = jnp.minimum(melt_pot, S_sn + P_sn)

    max_liquid = params.f_snlq_max * (S_sn + P_sn)
    possible_ret = max_liquid - S_snlq
    F_snlq = jnp.clip(R_sn_raw, 0.0, possible_ret)

    R_sn = R_sn_raw - F_snlq

    refreeze = jnp.where(T_srf < params.T_melt, S_snlq + F_snlq, 0.0)

    dS_snlq = F_snlq - refreeze
    S_snlq_new = jnp.clip(S_snlq + dS_snlq, 0.0, None)

    dS_sn = P_sn - R_sn_raw + refreeze
    S_sn_new = jnp.clip(S_sn + dS_sn, 0.0, None)

    return {
        "P_sn": jnp.maximum(0.0, P_sn),
        "R_sn": jnp.maximum(0.0, R_sn),
        "F_snlq": jnp.maximum(0.0, F_snlq),
        "refreeze": jnp.maximum(0.0, refreeze),
        "S_sn_new": jnp.maximum(0.0, S_sn_new),
        "S_snlq_new": jnp.maximum(0.0, S_snlq_new),
    }


def skin_canopy_module(
    P: jnp.ndarray,
    P_sn: jnp.ndarray,
    R_sn: jnp.ndarray,
    PET: jnp.ndarray,
    lu: LandUseParams,
    S_skin: jnp.ndarray,
    S_can: jnp.ndarray,
) -> Dict[str, Any]:
    P_ra = P - P_sn
    R_imp = P_ra * lu.imperv_frac
    P_ra_p = P_ra * (1 - lu.imperv_frac)
    out: Dict[str, Any] = {}

    f_skin = lu.canopy.f_bare * (1 - lu.imperv_frac)
    f_canv = lu.canopy.f_veg * (1 - lu.imperv_frac)

    for prc, f_lct, S_prc in [
        ("skin", f_skin, S_skin),
        ("can", f_canv, S_can),
    ]:
        S_norm = jnp.where(f_lct > 0.0, S_prc / f_lct, 0.0)
        S_max = lu.canopy.cap0 * (1.0 if prc == "skin" else lu.canopy.LAI)
        avail = jnp.maximum(0.0, S_norm + P_ra_p + R_sn)
        f_wet = jnp.where(S_max > 0.0, jnp.minimum(1.0, avail / S_max), 0.0)
        E_norm = jnp.clip(jnp.minimum(avail, PET * lu.crop_coeff * f_wet), 0.0, None)
        R_norm = jnp.maximum(0.0, avail - E_norm - S_max)
        S_norm_new = jnp.clip(avail - E_norm - R_norm, 0.0, S_max)
        out[f"S_{prc}_new"] = S_norm_new * f_lct
        out[f"E_{prc}_tot"] = E_norm * f_lct
        out[f"R_{prc}_tot"] = R_norm * f_lct

    out["R_imp"] = R_imp
    out["R_tr"] = out["R_skin_tot"] + out["R_can_tot"]
    return out


def soil_module(
    P: float,
    T_srf: float,
    PET: float,
    snow: dict,
    canopy: dict,
    lu: LandUseParams,
    S_so: float,
) -> Dict[str, Any]:
    params = lu.soil
    canopy_params = lu.canopy

    R_tr = canopy["R_tr"]
    E_can = canopy["E_can_tot"]
    E_skin = canopy["E_skin_tot"]
    f_bare = canopy_params.f_bare * (1 - lu.imperv_frac)
    f_veg = canopy_params.f_veg * (1 - lu.imperv_frac)

    S_so_sg = jnp.where(
        S_so > params.S_so_sg_min,
        params.S_so_sg_max
        - (params.S_so_sg_max - params.S_so_sg_min)
        * (1 - (S_so - params.S_so_sg_min) / (params.S_so_max - params.S_so_sg_min)) ** (1 / (1 + params.b)),
        S_so,
    )
    Δ_sg = params.S_so_sg_max - params.S_so_sg_min
    c1 = jnp.where(
        Δ_sg > 0,
        jnp.minimum(1.0, ((params.S_so_sg_max - S_so_sg) / Δ_sg) ** (1 + params.b)),
        1.0,
    )
    c2 = jnp.where(
        Δ_sg > 0,
        jnp.maximum(0.0, ((params.S_so_sg_max - S_so_sg - R_tr) / Δ_sg) ** (1 + params.b)),
        0.0,
    )

    is_frozen = T_srf < 273.15
    no_through = (R_tr <= 0) | (S_so_sg + R_tr <= params.S_so_sg_min)
    over_subgrid = S_so_sg + R_tr > params.S_so_sg_max

    ideal_infil = jnp.where(
        is_frozen,
        0.0,
        jnp.where(
            no_through,
            R_tr,
            jnp.where(
                over_subgrid,
                jnp.maximum(0.0, params.S_so_max - S_so),
                jnp.minimum(
                    R_tr,
                    jnp.maximum(0.0, S_so - params.S_so_max)
                    + (Δ_sg / (1 + params.b)) * (c1 - c2),
                ),
            ),
        ),
    )
    avail_cap = jnp.maximum(0.0, params.S_so_max - S_so)
    infiltration = jnp.clip(ideal_infil, 0.0, jnp.minimum(R_tr, avail_cap))
    R_srf = R_tr - infiltration

    PET_T = jnp.maximum(0.0, PET * lu.crop_coeff - E_can)
    PET_bs = jnp.maximum(0.0, PET * lu.crop_coeff - E_skin)

    theta_T = jnp.clip(
        (S_so - params.S_so_wilt) / (params.f_so_crit * params.S_so_max - params.S_so_wilt),
        0.0,
        1.0,
    )
    theta_bs = jnp.clip(
        (S_so - params.f_so_bs_low * params.S_so_max) / ((1 - params.f_so_bs_low) * params.S_so_max),
        0.0,
        1.0,
    )

    E_T_rel = PET_T * theta_T
    E_bs_rel = PET_bs * theta_bs

    E_T_cell = E_T_rel * f_veg
    E_bs_cell = E_bs_rel * f_bare

    R_gr_low = jnp.where(
        params.S_so_max > 0,
        params.R_gr_min * params.dt * (S_so / params.S_so_max),
        0.0,
    )
    frac2 = jnp.where(
        (params.S_so_max - params.S_so_grmax) > 0,
        jnp.maximum(0.0, S_so - params.S_so_grmax) / (params.S_so_max - params.S_so_grmax),
        0.0,
    )
    R_gr_high = jnp.where(
        S_so > params.S_so_grmax,
        (params.R_gr_max - params.R_gr_min) * params.dt * (frac2 ** params.R_gr_exp),
        0.0,
    )
    cond_low = (S_so <= params.S_so_grmin) | (T_srf < 273.15)
    cond_mid = S_so <= params.S_so_grmax
    R_gr = jnp.where(cond_low, 0.0, jnp.where(cond_mid, R_gr_low, R_gr_low + R_gr_high))

    dS_so = infiltration - R_gr - E_T_cell - E_bs_cell
    S_so_new = jnp.clip(S_so + dS_so, 0.0, params.S_so_max)

    return {
        "infiltration": infiltration,
        "R_srf": R_srf,
        "R_gr": R_gr,
        "E_T": E_T_cell,
        "E_bs": E_bs_cell,
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
) -> dict:
    P_sn = snow["P_sn"]
    R_sn = snow["R_sn"]
    R_srf = soil["R_srf"]
    R_gr = soil["R_gr"]

    f_sw = jnp.maximum(params.f_lake, params.f_wetland)
    P_ra = P - P_sn

    sw_inputs = (P_ra + R_sn) * f_sw + R_srf

    E_sw_potential = PET * f_sw
    E_sw = jnp.minimum(E_sw_potential, S_sw)

    R_sw = S_sw / (params.LAG_sw + 1.0)

    dS_sw = sw_inputs - E_sw - R_sw
    S_sw_new = jnp.clip(S_sw + dS_sw, 0.0, None)

    R_gw = S_gw / (params.LAG_gw + 1.0)

    dS_gw = R_gr - R_gw
    S_gw_new = jnp.clip(S_gw + dS_gw, 0.0, None)

    return {
        "f_sw": f_sw,
        "P_ra": P_ra,
        "R_srf": R_srf,
        "R_sw": R_sw,
        "E_sw": E_sw,
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
    P: jnp.ndarray,
    snow: dict,
    PET: jnp.ndarray,
    params: HydroParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    canopy = skin_canopy_module(P, snow["P_sn"], snow["R_sn"], PET, lu, state.S_skin, state.S_can)
    state = state._replace(
        S_skin=canopy["S_skin_new"],
        S_can=canopy["S_can_new"],
    )
    return state, canopy


def update_soil_state(
    state: HydroState,
    P: jnp.ndarray,
    T: jnp.ndarray,
    PET: jnp.ndarray,
    snow: dict,
    canopy: dict,
    params: HydroParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    soil = soil_module(P, T, PET, snow, canopy, lu, state.S_so)
    state = state._replace(S_so=soil["S_so_new"])
    return state, soil


def update_groundwater_state(
    state: HydroState,
    P: jnp.ndarray,
    PET: jnp.ndarray,
    snow: dict,
    soil: dict,
    params: HydroParams,
) -> Tuple[HydroState, Dict[str, Any]]:
    lu = params.landuse
    gw = groundwater_process(
        P=P,
        snow=snow,
        soil=soil,
        PET=PET,
        params=lu.groundwater,
        S_sw=state.S_sw,
        S_gw=state.groundwater,
    )
    state = state._replace(S_sw=gw["S_sw_new"], groundwater=gw["S_gw_new"])
    return state, gw


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
        state, gw = update_groundwater_state(state, p, e, snow, soil, params)
        runoff = gw["R_sw"] + gw["R_gw"]
        return state, runoff

    init = HydroState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    _, runoff = jax.lax.scan(step, init, (precip, evap, temp))
    return runoff


def build_cell_params(per_cell: Sequence[HydroParams]) -> HydroParams:
    _require_jax()
    return tree_map(lambda *xs: jnp.stack(xs), *per_cell)


def hydrologic_model(
    precip: jnp.ndarray,
    evap: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,
) -> jnp.ndarray:
    _require_jax()
    return jax.vmap(_single_cell_model, in_axes=(1, 1, 1, 0), out_axes=1)(precip, evap, temp, params)


__all__ = [
    "SnowParams",
    "CanopyParams",
    "SoilParams",
    "GroundwaterParams",
    "LandUseParams",
    "HydroParams",
    "HydroState",
    "build_cell_params",
    "hydrologic_model",
    "default_landuse_lookup",
]
