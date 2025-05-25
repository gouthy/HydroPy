"""Process-based hydrologic model implemented with JAX."""

from __future__ import annotations

from typing import NamedTuple

try:
    import jax.numpy as jnp
except ImportError as e:  # pragma: no cover - JAX must be installed
    raise ImportError(
        "JAX is required to use hydropy.model. Please install jax."
    ) from e


class SnowParams(NamedTuple):
    """Parameters controlling snow accumulation and melt."""

    melt_factor: float  # snowmelt per degree above freezing


def snow_process(precip: jnp.ndarray, temp: jnp.ndarray, params: SnowParams) -> jnp.ndarray:
    """Convert precipitation and temperature to liquid water input."""
    melt = jnp.maximum(temp, 0.0) * params.melt_factor
    return precip + melt


class CanopyParams(NamedTuple):
    """Parameters for canopy interception."""

    max_storage: float  # maximum canopy storage


def canopy_process(water: jnp.ndarray, params: CanopyParams) -> jnp.ndarray:
    """Simulate canopy interception and throughfall."""
    interception = jnp.minimum(water, params.max_storage)
    return water - interception


class SoilParams(NamedTuple):
    """Parameters for infiltration and surface runoff."""

    infiltration_rate: float
    field_capacity: float


def soil_process(water: jnp.ndarray, params: SoilParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Partition water into infiltration and surface runoff."""
    infiltration = jnp.minimum(water, params.infiltration_rate)
    runoff = jnp.maximum(water - params.field_capacity, 0.0)
    return infiltration, runoff


class GroundwaterParams(NamedTuple):
    """Parameters for shallow groundwater flow."""

    baseflow_coeff: float


def groundwater_process(recharge: jnp.ndarray, params: GroundwaterParams) -> jnp.ndarray:
    """Compute baseflow from groundwater recharge."""
    return params.baseflow_coeff * recharge


class HydroParams(NamedTuple):
    """Collection of parameters for the hydrologic model."""

    snow: SnowParams
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams


def hydrologic_model(precip: jnp.ndarray, temp: jnp.ndarray, params: HydroParams) -> jnp.ndarray:
    """Run the simple process-based hydrologic model."""
    water = snow_process(precip, temp, params.snow)
    throughfall = canopy_process(water, params.canopy)
    infiltration, runoff_surface = soil_process(throughfall, params.soil)
    baseflow = groundwater_process(infiltration, params.groundwater)
    return runoff_surface + baseflow


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
