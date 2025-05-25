
"""Distributed hydrologic model built with JAX."""

from __future__ import annotations

from typing import NamedTuple

try:
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError(
        "JAX is required to use hydropy.model. Please install jax."
    ) from e


class SnowParams(NamedTuple):
    """Parameters controlling snow accumulation and melt."""

    melt_temp: float = 0.0
    melt_rate: float = 1.0


class CanopyParams(NamedTuple):
    """Parameters for the canopy interception process."""

    capacity: float = 1.0
    drip_coeff: float = 0.1
    evaporation_coeff: float = 1.0


class SoilParams(NamedTuple):
    """Parameters for soil water balance."""

    capacity: float = 2.0
    percolation_rate: float = 0.5
    evap_coeff: float = 1.0


class GroundwaterParams(NamedTuple):
    """Parameters for groundwater storage and release."""

    recession_coeff: float = 0.3


class HydroParams(NamedTuple):
    """Container for all process parameters."""
    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


def snow_process(
    precip: jnp.ndarray, temp: jnp.ndarray, state: jnp.ndarray, params: SnowParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Partition precipitation into rain and snow and melt existing snow."""
    snowfall = jnp.where(temp <= params.melt_temp, precip, 0.0)
    rain = jnp.where(temp > params.melt_temp, precip, 0.0)
    snowpack = state + snowfall
    melt = jnp.where(temp > params.melt_temp, params.melt_rate * (temp - params.melt_temp), 0.0)
    melt = jnp.minimum(melt, snowpack)
    snowpack = snowpack - melt
    liquid = rain + melt
    return liquid, snowpack


def canopy_process(
    water_in: jnp.ndarray,
    evap: jnp.ndarray,
    state: jnp.ndarray,
    params: CanopyParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Store water in the canopy and pass the remainder to the ground."""
    canopy = state + water_in
    canopy = jnp.clip(canopy, 0.0, params.capacity)
    throughfall = jnp.maximum(state + water_in - params.capacity, 0.0)
    evap_loss = jnp.minimum(evap * params.evaporation_coeff, canopy)
    canopy = canopy - evap_loss
    drip = params.drip_coeff * canopy
    canopy = canopy - drip
    throughfall = throughfall + drip
    return throughfall, canopy


def soil_process(
    water_in: jnp.ndarray,
    evap: jnp.ndarray,
    state: jnp.ndarray,
    params: SoilParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Update soil water and generate percolation."""
    soil = state + water_in
    soil = jnp.clip(soil, 0.0, params.capacity)
    percolation = params.percolation_rate * soil
    soil = soil - percolation
    evap_loss = jnp.minimum(evap * params.evap_coeff, soil)
    soil = soil - evap_loss
    return percolation, soil


def groundwater_process(
    recharge: jnp.ndarray, state: jnp.ndarray, params: GroundwaterParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate runoff from groundwater storage."""
    gw = state + recharge
    runoff = params.recession_coeff * gw
    gw = gw - runoff
    return runoff, gw


def hydrologic_model(
    precip: jnp.ndarray,
    temp: jnp.ndarray,
    evap: jnp.ndarray,
    params: HydroParams,
) -> jnp.ndarray:
    """Run the distributed hydrologic model over all locations."""
    time_steps, nloc = precip.shape
    snow_state = jnp.zeros(nloc)
    canopy_state = jnp.zeros(nloc)
    soil_state = jnp.zeros(nloc)
    gw_state = jnp.zeros(nloc)
    out = []
    for t in range(time_steps):
        liq, snow_state = snow_process(precip[t], temp[t], snow_state, params.snow)
        infil, canopy_state = canopy_process(liq, evap[t], canopy_state, params.canopy)
        recharge, soil_state = soil_process(infil, evap[t], soil_state, params.soil)
        runoff, gw_state = groundwater_process(recharge, gw_state, params.groundwater)
        out.append(runoff)
    return jnp.stack(out)



__all__ = [
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
