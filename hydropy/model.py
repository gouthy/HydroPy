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
    melt_temp: float
    melt_rate: float


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
    snowpack: float
    canopy: float
    soil: float
    groundwater: float


def update_snow_state(
    state: HydroState, precip: float, temp: float, params: HydroParams
) -> tuple[HydroState, float]:
    """Update snow state and return rainfall produced."""
    rainfall, snowpack = snow_process(state.snowpack, precip, temp, params.snow)
    return state._replace(snowpack=snowpack), rainfall


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


def snow_process(snowpack: float, precip: float, temp: float, params: SnowParams):
    snowfall = jnp.where(temp <= params.melt_temp, precip, 0.0)
    rainfall = jnp.where(temp > params.melt_temp, precip, 0.0)
    melt = jnp.where(
        temp > params.melt_temp,
        jnp.minimum(snowpack, params.melt_rate * (temp - params.melt_temp)),
        0.0,
    )
    rainfall = rainfall + melt
    new_snow = snowpack + snowfall - melt
    return rainfall, new_snow


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

    init = HydroState(0.0, 0.0, 0.0, 0.0)
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
