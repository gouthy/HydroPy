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
        snow=SnowParams(
            t_sn_min=272.05,
            t_sn_max=276.45,
            t_melt=273.15,
            f_snlq_max=0.06,
        ),
        canopy=CanopyParams(capacity=1.0),
        soil=SoilParams(capacity=2.0, conductivity=0.1),
        groundwater=GroundwaterParams(recession=0.05),
    )
    runoff = hydrologic_model(precip, evap, temp, params)
    assert runoff.shape == precip.shape
