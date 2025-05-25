"""Collection of simple JAX-based hydrologic process models."""

from __future__ import annotations

from typing import NamedTuple

try:
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError(
        "JAX is required to use hydropy.model. Please install jax."
    ) from e


class SnowParams(NamedTuple):
    """Parameters for the snow process."""

    melt_temp: float


def snow_process(precip: jnp.ndarray, temp: jnp.ndarray, params: SnowParams) -> jnp.ndarray:
    """Melt snow to produce liquid precipitation."""
    return jnp.where(temp > params.melt_temp, precip, 0.0)


class CanopyParams(NamedTuple):
    """Parameters for the canopy process."""

    capacity: float


def canopy_process(precip: jnp.ndarray, params: CanopyParams) -> jnp.ndarray:
    """Interception by vegetation canopy."""
    interception = jnp.minimum(precip, params.capacity)
    return precip - interception


class SoilParams(NamedTuple):
    """Parameters for the soil process."""

    capacity: float
    evap_coeff: float


def soil_process(throughfall: jnp.ndarray, evap: jnp.ndarray, state: float, params: SoilParams) -> tuple[float, jnp.ndarray]:
    """Update soil moisture and return groundwater recharge."""
    storage = jnp.clip(state + throughfall - params.evap_coeff * evap, 0.0, params.capacity)
    recharge = jnp.maximum(storage - params.capacity, 0.0)
    storage = storage - recharge
    return float(storage), recharge


class GroundwaterParams(NamedTuple):
    """Parameters for the groundwater process."""

    outflow_coeff: float


def groundwater_process(recharge: jnp.ndarray, state: float, params: GroundwaterParams) -> tuple[float, jnp.ndarray]:
    """Update groundwater storage and produce runoff."""
    storage = state + recharge
    runoff = params.outflow_coeff * storage
    storage = storage - runoff
    return float(storage), runoff


class HydroParams(NamedTuple):
    """Container for all process parameters."""

    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


def hydrologic_model(
    precip: jnp.ndarray,
    temp: jnp.ndarray,
    evap: jnp.ndarray,
    params: HydroParams,
) -> jnp.ndarray:
    """Simulate runoff from meteorological inputs."""
    soil_state = 0.0
    gw_state = 0.0
    runoff_series = []
    for p, t, e in zip(precip, temp, evap):
        melt = snow_process(p, t, params.snow)
        throughfall = canopy_process(melt, params.canopy)
        soil_state, recharge = soil_process(throughfall, e, soil_state, params.soil)
        gw_state, runoff = groundwater_process(recharge, gw_state, params.groundwater)
        runoff_series.append(runoff)
    return jnp.stack(runoff_series)


__all__ = [
    "SnowParams",
    "CanopyParams",
    "SoilParams",
    "GroundwaterParams",
    "HydroParams",
    "hydrologic_model",
]
