import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from hydropy import (
    RRParams,
    rainfall_runoff,
    SnowParams,
    CanopyParams,
    SoilParams,
    GroundwaterParams,
    HydroParams,
    hydrologic_model,
)


def test_runoff_shape():
    precip = jnp.array([1.0, 0.0, 0.5])
    evap = jnp.array([0.2, 0.2, 0.2])
    params = RRParams(capacity=1.0, evap_coeff=1.0)
    runoff = rainfall_runoff(precip, evap, params)
    assert runoff.shape == precip.shape


def test_hydrologic_model_shape():
    precip = jnp.ones((3, 2))
    temp = jnp.zeros((3, 2))
    evap = jnp.full((3, 2), 0.1)
    params = HydroParams(
        snow=SnowParams(),
        canopy=CanopyParams(),
        soil=SoilParams(),
        groundwater=GroundwaterParams(),
    )
    runoff = hydrologic_model(precip, temp, evap, params)
    assert runoff.shape == precip.shape

