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
    temp = jnp.array([2.0, -1.0, 0.0])
    params = HydroParams(
        snow=SnowParams(melt_factor=0.1),
        canopy=CanopyParams(max_storage=0.1),
        soil=SoilParams(infiltration_rate=0.2, field_capacity=0.5),
        groundwater=GroundwaterParams(baseflow_coeff=0.3),
    )
    runoff = hydrologic_model(precip, temp, params)
    assert runoff.shape == precip.shape

