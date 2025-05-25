import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from hydropy import (
    RRParams,
    rainfall_runoff,
    HydroParams,
    SnowParams,
    CanopyParams,
    SoilParams,
    GroundwaterParams,
    hydrologic_model,
)


def test_runoff_shape():
    precip = jnp.array([1.0, 0.0, 0.5])
    evap = jnp.array([0.2, 0.2, 0.2])
    params = RRParams(capacity=1.0, evap_coeff=1.0)
    runoff = rainfall_runoff(precip, evap, params)
    assert runoff.shape == precip.shape


def test_hydrologic_model_shape():
    precip = jnp.array([1.0, 0.0, 0.5])
    temp = jnp.array([1.0, -1.0, 2.0])
    params = HydroParams(
        snow=SnowParams(melt_temp=0.0, melt_rate=1.0),
        canopy=CanopyParams(capacity=0.2),
        soil=SoilParams(field_capacity=1.0, percolation_rate=0.1),
        groundwater=GroundwaterParams(baseflow_coeff=0.05),
    )
    runoff = hydrologic_model(precip, temp, params)
    assert runoff.shape == precip.shape

