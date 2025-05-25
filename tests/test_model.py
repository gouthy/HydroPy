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
    precip = jnp.ones((3, 2)) * 0.5
    evap = jnp.ones((3, 2)) * 0.1
    temp = jnp.ones((3, 2))
    params = HydroParams(
        snow=SnowParams(melt_temp=0.0, melt_rate=0.1),
        canopy=CanopyParams(capacity=1.0),
        soil=SoilParams(capacity=2.0, conductivity=0.1),
        groundwater=GroundwaterParams(recession=0.05),
    )
    runoff = hydrologic_model(precip, evap, temp, params)
    assert runoff.shape == precip.shape
