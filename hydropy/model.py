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
    """Parameters controlling the snow process."""

    melt_temp: float = 0.0
    melt_coeff: float = 1.0


class CanopyParams(NamedTuple):
    """Parameters for canopy interception and evaporation."""

    capacity: float = 1.0
    evap_coeff: float = 1.0


class SoilParams(NamedTuple):
    """Parameters describing soil moisture behaviour."""

    capacity: float = 1.0
    evap_coeff: float = 1.0


class GroundwaterParams(NamedTuple):
    """Parameters for a simple linear groundwater store."""

    recession_coeff: float = 0.5


class HydroParams(NamedTuple):
    """Grouped parameters for the full hydrologic model."""

    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


def _snow_step(precip: jnp.ndarray, temp: jnp.ndarray, snow: jnp.ndarray, params: SnowParams):
    """Very simple degree-day snow model."""

    melt = jnp.where(temp > params.melt_temp, params.melt_coeff * (temp - params.melt_temp), 0.0)
    snow = jnp.maximum(snow + precip - melt, 0.0)
    return snow, melt


def _canopy_step(precip: jnp.ndarray, evap: jnp.ndarray, canopy: jnp.ndarray, params: CanopyParams):
    """Simple canopy interception and evaporation."""

    intercepted = jnp.minimum(precip, params.capacity - canopy)
    canopy = canopy + intercepted
    throughfall = precip - intercepted
    evap_loss = params.evap_coeff * evap
    canopy = jnp.maximum(canopy - evap_loss, 0.0)
    return canopy, throughfall


def _soil_step(water: jnp.ndarray, evap: jnp.ndarray, soil: jnp.ndarray, params: SoilParams):
    """Update soil moisture and compute percolation."""

    soil = jnp.minimum(soil + water - params.evap_coeff * evap, params.capacity)
    percolation = jnp.maximum(soil - params.capacity, 0.0)
    soil = soil - percolation
    return soil, percolation


def _groundwater_step(recharge: jnp.ndarray, gw: jnp.ndarray, params: GroundwaterParams):
    """Linear reservoir for groundwater."""

    gw = gw + recharge
    baseflow = params.recession_coeff * gw
    gw = gw - baseflow
    return gw, baseflow


def hydrologic_model(
    precip: jnp.ndarray,
    temp: jnp.ndarray,
    evap: jnp.ndarray,
    params: HydroParams,
) -> jnp.ndarray:
    """Run a distributed, process-based hydrologic model.

    Args:
        precip: Precipitation array with shape ``(time, n_cells)``.
        temp: Temperature array with shape ``(time, n_cells)``.
        evap: Potential evapotranspiration with shape ``(time, n_cells)``.
        params: Model parameters.

    Returns:
        Runoff array of shape ``(time, n_cells)``.
    """

    n_cells = precip.shape[1]
    snow = jnp.zeros(n_cells)
    canopy = jnp.zeros(n_cells)
    soil = jnp.zeros(n_cells)
    gw = jnp.zeros(n_cells)

    runoff = []
    for p, t, e in zip(precip, temp, evap):
        snow, melt = _snow_step(p, t, snow, params.snow)
        canopy, throughfall = _canopy_step(melt, e, canopy, params.canopy)
        soil, recharge = _soil_step(throughfall, e, soil, params.soil)
        gw, baseflow = _groundwater_step(recharge, gw, params.groundwater)
        runoff.append(baseflow)

    return jnp.stack(runoff)


__all__ = [
    "RRParams",
    "rainfall_runoff",
    "SnowParams",
    "CanopyParams",
    "SoilParams",
    "GroundwaterParams",
    "HydroParams",
    "hydrologic_model",
]
