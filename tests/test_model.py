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
    params = HydroParams(
        snow=SnowParams(melt_temp=0.0, melt_rate=1.0),
        canopy=CanopyParams(capacity=1.0, evap_coeff=0.1),
        soil=SoilParams(field_capacity=1.0, percolation_coeff=0.1),
        groundwater=GroundwaterParams(baseflow_coeff=0.05),
    )
    runoff = hydrologic_model(precip, temp, params)
    assert runoff.shape == precip.shape
