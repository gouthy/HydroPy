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


def test_runoff_shape():
    precip = jnp.array([1.0, 0.0, 0.5])
    temp = jnp.array([1.0, 1.0, 1.0])
    evap = jnp.array([0.2, 0.2, 0.2])
    params = HydroParams(
        snow=SnowParams(melt_temp=0.0),
        canopy=CanopyParams(capacity=0.1),
        soil=SoilParams(capacity=1.0, evap_coeff=1.0),
        groundwater=GroundwaterParams(outflow_coeff=0.5),
    )
    runoff = hydrologic_model(precip, temp, evap, params)
    assert runoff.shape == precip.shape

