import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from hydropy import (
    SnowParams,
    CanopyParams,
    SoilParams,
    GroundwaterParams,
    HydroParams,
    hydrologic_model,
)


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
