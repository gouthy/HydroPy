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
    """Parameters for the snow module."""

    t_sn_min: float
    """Lower bound for snowfall diagnosis [K]."""

    t_sn_max: float
    """Upper bound for snowfall diagnosis [K]."""

    t_melt: float
    """Melting temperature [K]."""

    f_snlq_max: float
    """Maximum liquid snow fraction [–]."""


class CanopyParams(NamedTuple):
    """Parameters controlling skin and canopy interception."""

    f_bare: float
    f_veg: float
    LAI: float
    cap0: float


class SoilParams(NamedTuple):
    capacity: float
    conductivity: float


class GroundwaterParams(NamedTuple):
    recession: float


class HydroParams(NamedTuple):
    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


class HydroState(NamedTuple):
    """Current model state."""

    snow_solid: float
    """Solid snow storage [mm]."""

    snow_liquid: float
    """Liquid water in the snowpack [mm]."""

    skin: float
    canopy: float
    soil: float
    groundwater: float


# ---------------------------------------------------------------------------
# State update helpers
# ---------------------------------------------------------------------------

def update_snow_state(
    state: HydroState,
    precip: float,
    temp: float,
    params: HydroParams,
    day_fraction: float = 1.0,
) -> tuple[HydroState, float, float, float]:
    """Update snow state and return rainfall, snowfall and melt."""
    rain, snow_solid, snow_liquid, p_sn, r_sn = snow_process(
        state.snow_solid,
        state.snow_liquid,
        precip,
        temp,
        day_fraction,
        params.snow,
    )
    state = state._replace(snow_solid=snow_solid, snow_liquid=snow_liquid)
    return state, rain, p_sn, r_sn


def update_skin_canopy_state(
    state: HydroState,
    precip: float,
    p_sn: float,
    r_sn: float,
    evap: float,
    params: HydroParams,
) -> tuple[HydroState, float]:
    """Update skin and canopy stores and return throughfall."""
    throughfall, skin, canopy = skin_canopy_process(
        state.skin,
        state.canopy,
        precip,
        p_sn,
        r_sn,
        evap,
        params.canopy,
    )
    state = state._replace(skin=skin, canopy=canopy)
    return state, throughfall


def update_soil_state(
    state: HydroState, water: float, params: HydroParams
) -> tuple[HydroState, float, float]:
    """Update soil state and return surface runoff and recharge."""
    surface_runoff, recharge, soil = soil_process(state.soil, water, params.soil)
    return state._replace(soil=soil), surface_runoff, recharge


def update_groundwater_state(
    state: HydroState, recharge: float, params: HydroParams
) -> tuple[HydroState, float]:
    """Update groundwater state and return baseflow."""
    baseflow, gw = groundwater_process(state.groundwater, recharge, params.groundwater)
    return state._replace(groundwater=gw), baseflow


# ---------------------------------------------------------------------------
# Process representations
# ---------------------------------------------------------------------------

def snow_process(
    snow_solid: float,
    snow_liquid: float,
    precip: float,
    temp: float,
    day_fraction: float,
    params: SnowParams,
) -> tuple[float, float, float, float, float]:
    """Snow accumulation and melt.

    Returns rainfall, updated solid & liquid stores, snowfall, and melt.
    """
    frac_snow = (params.t_sn_max - temp) / (params.t_sn_max - params.t_sn_min)
    frac_snow = jnp.clip(frac_snow, 0.0, 1.0)
    p_sn = precip * frac_snow

    DDF = day_fraction * 8.3 + 0.7
    melt_pot = DDF * (temp - params.t_melt)
    melt_pot = jnp.maximum(0.0, melt_pot)

    r_sn_raw = jnp.minimum(melt_pot, snow_solid + p_sn)

    max_liquid = params.f_snlq_max * (snow_solid + p_sn)
    possible_retention = max_liquid - snow_liquid
    f_snlq = jnp.clip(r_sn_raw, 0.0, possible_retention)
    r_sn = r_sn_raw - f_snlq

    snow_liquid_new = snow_liquid + f_snlq
    refreeze = jnp.where(temp < params.t_melt, snow_liquid_new, 0.0)
    snow_liquid_new = jnp.where(temp < params.t_melt, 0.0, snow_liquid_new)

    delta_solid = p_sn - r_sn + refreeze
    snow_solid_new = snow_solid + delta_solid

    rainfall = (precip - p_sn) + r_sn

    return rainfall, snow_solid_new, snow_liquid_new, p_sn, r_sn


def skin_canopy_process(
    skin: float,
    canopy: float,
    precip: float,
    p_sn: float,
    r_sn: float,
    evap: float,
    params: CanopyParams,
) -> tuple[float, float, float]:
    """Skin and canopy water balance."""
    p_ra = precip - p_sn

    def _component(storage, frac, lai):
        if frac <= 0.0:
            return 0.0, 0.0, 0.0
        s_norm = storage / frac
        s_max = params.cap0 * lai
        avail = s_norm + p_ra + r_sn
        f_wet = jnp.where(s_max > 0.0, jnp.minimum(1.0, avail / s_max), 0.0)
        e_norm = jnp.minimum(avail, evap * f_wet)
        r_norm = jnp.maximum(0.0, avail - e_norm - s_max)
        s_norm_new = jnp.minimum(s_max, avail - e_norm)
        return s_norm_new * frac, e_norm * frac, r_norm * frac

    s_skin_new, _e_skin, r_skin = _component(skin, params.f_bare, 1.0)
    s_can_new, _e_can, r_can = _component(canopy, params.f_veg, params.LAI)
    r_tr = r_skin + r_can
    return r_tr, s_skin_new, s_can_new


def soil_process(soil: float, water: float, params: SoilParams):
    infil = jnp.minimum(params.capacity - soil, water)
    surface_runoff = water - infil
    percolation = jnp.minimum(soil + infil, params.conductivity)
    soil = soil + infil - percolation
    return surface_runoff, percolation, soil


def groundwater_process(gw: float, recharge: float, params: GroundwaterParams):
    baseflow = params.recession * gw
    gw = gw + recharge - baseflow
    return baseflow, gw


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

def _single_cell_model(
    precip: jnp.ndarray, evap: jnp.ndarray, temp: jnp.ndarray, params: HydroParams
) -> jnp.ndarray:
    _require_jax()

    def step(state: HydroState, inputs):
        p, e, t = inputs
        state, _rain, p_sn, r_sn = update_snow_state(state, p, t, params)
        state, tf = update_skin_canopy_state(state, p, p_sn, r_sn, e, params)
        state, surf, recharge = update_soil_state(state, tf, params)
        state, base = update_groundwater_state(state, recharge, params)
        runoff = surf + base
        return state, runoff

    init = HydroState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    _, runoff = jax.lax.scan(step, init, (precip, evap, temp))
    return runoff


def hydrologic_model(
    precip: jnp.ndarray, evap: jnp.ndarray, temp: jnp.ndarray, params: HydroParams
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
