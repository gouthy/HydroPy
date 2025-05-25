"""Simple JAX-based hydrologic model."""

from __future__ import annotations

from typing import NamedTuple

try:
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError(
        "JAX is required to use hydropy.model. Please install jax." 
    ) from e


class RRParams(NamedTuple):
    """Parameters for the rainfall-runoff model."""

    capacity: float
    evap_coeff: float


def rainfall_runoff(precip: jnp.ndarray, evap: jnp.ndarray, params: RRParams) -> jnp.ndarray:
    """Compute runoff from precipitation and evapotranspiration.

    Args:
        precip: Precipitation time series.
        evap: Potential evapotranspiration time series.
        params: Model parameters.

    Returns:
        Runoff time series.
    """
    # Simple bucket model
    storage = 0.0
    runoff = []
    for p, e in zip(precip, evap):
        storage = jnp.clip(storage + p - params.evap_coeff * e, 0.0, params.capacity)
        r = jnp.maximum(storage - params.capacity, 0.0)
        storage = storage - r
        runoff.append(r)
    return jnp.stack(runoff)



class SnowParams(NamedTuple):
    """Parameters for the snow process."""

    melt_temp: float
    melt_rate: float


class CanopyParams(NamedTuple):
    """Parameters for the canopy interception process."""

    capacity: float


class SoilParams(NamedTuple):
    """Parameters controlling the soil bucket."""

    field_capacity: float
    percolation_rate: float


class GroundwaterParams(NamedTuple):
    """Parameters for groundwater storage and baseflow."""

    baseflow_coeff: float


class HydroParams(NamedTuple):
    """Aggregate container for the full model parameters."""

    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


def snow_process(
    precip: jnp.ndarray, temp: jnp.ndarray, params: SnowParams
) -> jnp.ndarray:
    """Simulate accumulation and melt of snow."""

    snowpack = 0.0
    liquid = []
    for p, t in zip(precip, temp):
        snowfall = jnp.where(t <= params.melt_temp, p, 0.0)
        rainfall = jnp.where(t > params.melt_temp, p, 0.0)
        snowpack = snowpack + snowfall
        melt = jnp.where(t > params.melt_temp, params.melt_rate * (t - params.melt_temp), 0.0)
        melt = jnp.clip(melt, 0.0, snowpack)
        snowpack = snowpack - melt
        liquid.append(rainfall + melt)
    return jnp.stack(liquid)


def canopy_process(water: jnp.ndarray, params: CanopyParams) -> jnp.ndarray:
    """Interception by vegetation canopy."""

    storage = 0.0
    throughfall = []
    for w in water:
        storage = storage + w
        tf = jnp.maximum(storage - params.capacity, 0.0)
        storage = jnp.minimum(storage, params.capacity)
        throughfall.append(tf)
    return jnp.stack(throughfall)


def soil_process(
    water: jnp.ndarray, params: SoilParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Soil storage and percolation to groundwater."""

    storage = 0.0
    recharge = []
    runoff = []
    for w in water:
        storage = storage + w
        percolation = params.percolation_rate * storage
        storage = jnp.clip(storage - percolation, 0.0, params.field_capacity)
        direct_runoff = jnp.maximum(storage - params.field_capacity, 0.0)
        storage = storage - direct_runoff
        recharge.append(percolation)
        runoff.append(direct_runoff)
    return jnp.stack(recharge), jnp.stack(runoff)


def groundwater_process(
    recharge: jnp.ndarray, params: GroundwaterParams
) -> jnp.ndarray:
    """Convert recharge to baseflow."""

    gw = 0.0
    baseflow = []
    for r in recharge:
        gw = gw + r
        bf = gw * params.baseflow_coeff
        gw = gw - bf
        baseflow.append(bf)
    return jnp.stack(baseflow)


def hydrologic_model(
    precip: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,
) -> jnp.ndarray:
    """Full physically inspired rainfall-runoff model."""

    liquid = snow_process(precip, temp, params.snow)
    throughfall = canopy_process(liquid, params.canopy)
    recharge, surf = soil_process(throughfall, params.soil)
    base = groundwater_process(recharge, params.groundwater)
    return surf + base


__all__ = [
    "RRParams",
    "rainfall_runoff",
    "SnowParams",
    "CanopyParams",
    "SoilParams",
    "GroundwaterParams",
    "HydroParams",
    "snow_process",
    "canopy_process",
    "soil_process",
    "groundwater_process",
    "hydrologic_model",
]
