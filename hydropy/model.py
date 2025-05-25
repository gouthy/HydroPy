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
    capacity: float


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

    canopy: float
    soil: float
    groundwater: float


def update_snow_state(
    state: HydroState,
    precip: float,
    temp: float,
    params: HydroParams,
    day_fraction: float = 1.0,
) -> tuple[HydroState, float]:
    """Update snow state and return rainfall produced."""
    rainfall, sn_solid, sn_liquid = snow_process(
        state.snow_solid, state.snow_liquid, precip, temp, day_fraction, params.snow

    )
    return state._replace(snow_solid=sn_solid, snow_liquid=sn_liquid), rainfall


def update_canopy_state(
    state: HydroState, rainfall: float, evap: float, params: HydroParams
) -> tuple[HydroState, float]:
    """Update canopy state and return throughfall."""
    throughfall, canopy = canopy_process(state.canopy, rainfall, evap, params.canopy)
    return state._replace(canopy=canopy), throughfall


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


def snow_process(
    snow_solid: float,
    snow_liquid: float,
    precip: float,
    temp: float,
    day_fraction: float,
    params: SnowParams,
) -> tuple[float, float, float]:
    """Snow accumulation and melt.

    Returns rainfall leaving the snowpack and updated solid and liquid stores.
    """

    # 1) diagnose snowfall fraction
    frac_snow = (params.t_sn_max - temp) / (params.t_sn_max - params.t_sn_min)
    frac_snow = jnp.clip(frac_snow, 0.0, 1.0)
    P_sn = precip * frac_snow

    # 2) potential melt using degree day factor
    DDF = day_fraction * 8.3 + 0.7
    melt_pot = DDF * (temp - params.t_melt)
    melt_pot = jnp.maximum(0.0, melt_pot)

    R_sn_raw = jnp.minimum(melt_pot, snow_solid + P_sn)

    # 3) retain some melt as liquid water
    max_liquid = params.f_snlq_max * (snow_solid + P_sn)
    possible_retention = max_liquid - snow_liquid
    F_snlq = jnp.clip(R_sn_raw, 0.0, possible_retention)
    R_sn = R_sn_raw - F_snlq

    # 4) update liquid snow storage
    S_snlq_new = snow_liquid + F_snlq

    # 5) refreezing if below melting temperature
    refreeze = jnp.where(temp < params.t_melt, S_snlq_new, 0.0)
    S_snlq_new = jnp.where(temp < params.t_melt, 0.0, S_snlq_new)

    # 6) update solid snow storage
    delta_solid = P_sn - R_sn + refreeze
    S_sn_new = snow_solid + delta_solid

    rainfall = (precip - P_sn) + R_sn

    return rainfall, S_sn_new, S_snlq_new


def canopy_process(canopy: float, rainfall: float, evap: float, params: CanopyParams):
    intercept = jnp.minimum(params.capacity - canopy, rainfall)
    canopy = canopy + intercept
    throughfall = rainfall - intercept
    evap_canopy = jnp.minimum(canopy, evap)
    canopy = canopy - evap_canopy
    return throughfall, canopy


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


def _single_cell_model(precip: jnp.ndarray, evap: jnp.ndarray, temp: jnp.ndarray, params: HydroParams) -> jnp.ndarray:
    _require_jax()
    def step(state: HydroState, inputs):
        p, e, t = inputs
        state, rain = update_snow_state(state, p, t, params)
        state, tf = update_canopy_state(state, rain, e, params)
        state, surf, recharge = update_soil_state(state, tf, params)
        state, base = update_groundwater_state(state, recharge, params)
        runoff = surf + base
        return state, runoff

    init = HydroState(0.0, 0.0, 0.0, 0.0, 0.0)
    _, runoff = jax.lax.scan(step, init, (precip, evap, temp))
    return runoff


def hydrologic_model(precip: jnp.ndarray, evap: jnp.ndarray, temp: jnp.ndarray, params: HydroParams) -> jnp.ndarray:
    """Run distributed hydrologic model.

    Parameters are shared across all locations.
    Inputs are arrays of shape (time, n_locations).
    """
    _require_jax()
    return jax.vmap(_single_cell_model, in_axes=(1,1,1,None), out_axes=1)(precip, evap, temp, params)



__all__ = [
    "SnowParams",
    "CanopyParams",
    "SoilParams",
    "GroundwaterParams",
    "HydroParams",
    "hydrologic_model",
]
