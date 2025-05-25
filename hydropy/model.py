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
    """Parameters controlling snow processes."""

    melt_temp: float  # temperature threshold for melt
    melt_rate: float  # melt factor per degree above threshold


class CanopyParams(NamedTuple):
    """Parameters for interception and canopy evaporation."""

    capacity: float
    evap_coeff: float


class SoilParams(NamedTuple):
    """Parameters for soil moisture processes."""

    field_capacity: float
    percolation_coeff: float


class GroundwaterParams(NamedTuple):
    """Parameters for groundwater storage."""

    baseflow_coeff: float


class HydroParams(NamedTuple):
    """Container for all model parameters."""

    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


def hydrologic_model(
    precip: jnp.ndarray, temp: jnp.ndarray, params: HydroParams
) -> jnp.ndarray:
    """Simulate runoff for multiple locations.

    Args:
        precip: Precipitation array ``[time, n_cells]``.
        temp: Temperature array ``[time, n_cells]``.
        params: Model parameters.

    Returns:
        Runoff array with shape ``[time, n_cells]``.
    """
    snow_storage = jnp.zeros(precip.shape[1])
    canopy_storage = jnp.zeros_like(snow_storage)
    soil_storage = jnp.zeros_like(snow_storage)
    gw_storage = jnp.zeros_like(snow_storage)
    runoff_ts = []

    for p, t in zip(precip, temp):
        # Snow processes
        snow_storage = snow_storage + jnp.where(t < params.snow.melt_temp, p, 0.0)
        melt = jnp.where(
            t >= params.snow.melt_temp,
            params.snow.melt_rate * (t - params.snow.melt_temp),
            0.0,
        )
        melt = jnp.minimum(melt, snow_storage)
        snow_storage = snow_storage - melt
        input_water = jnp.where(t >= params.snow.melt_temp, p, 0.0) + melt

        # Canopy processes
        canopy_storage = canopy_storage + input_water
        evap = params.canopy.evap_coeff * canopy_storage
        canopy_storage = jnp.clip(canopy_storage - evap, 0.0, params.canopy.capacity)
        throughfall = jnp.maximum(canopy_storage - params.canopy.capacity, 0.0)
        canopy_storage = canopy_storage - throughfall

        # Soil processes
        soil_storage = soil_storage + throughfall
        percolation = params.soil.percolation_coeff * soil_storage
        soil_storage = jnp.clip(
            soil_storage - percolation, 0.0, params.soil.field_capacity
        )
        runoff_soil = jnp.maximum(soil_storage - params.soil.field_capacity, 0.0)
        soil_storage = soil_storage - runoff_soil

        # Groundwater processes
        gw_storage = gw_storage + percolation
        baseflow = params.groundwater.baseflow_coeff * gw_storage
        gw_storage = gw_storage - baseflow

        runoff_ts.append(runoff_soil + baseflow)

    return jnp.stack(runoff_ts)


__all__ = [
    "SnowParams",
    "CanopyParams",
    "SoilParams",
    "GroundwaterParams",
    "HydroParams",
    "hydrologic_model",
]
