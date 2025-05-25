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
        rain, snow = snow_process(state.snowpack, p, t, params.snow)
        tf, canopy = canopy_process(state.canopy, rain, e, params.canopy)
        surf, recharge, soil = soil_process(state.soil, tf, params.soil)
        base, gw = groundwater_process(state.groundwater, recharge, params.groundwater)
        new_state = HydroState(snow, canopy, soil, gw)
        runoff = surf + base
        return new_state, runoff

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
