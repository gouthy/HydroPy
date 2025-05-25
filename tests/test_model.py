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
    pet = jnp.ones((3, 2)) * 0.1
    temp = jnp.ones((3, 2)) * 273.15
    params = HydroParams(
        snow=SnowParams(),
        canopy=CanopyParams(f_bare=0.5, f_veg=0.5, LAI=2.0, cap0=1.0),
        soil=SoilParams(
            S_so_max=100.0,
            S_so_wilt=5.0,
            S_so_grmin=10.0,
            S_so_grmax=50.0,
            S_so_sg_min=10.0,
            S_so_sg_max=100.0,
            b=1.0,
            R_gr_min=0.01,
            R_gr_max=0.1,
            dt=86400.0,
        ),
        groundwater=GroundwaterParams(f_lake=0.1, f_wetland=0.2, LAG_sw=2.0, LAG_gw=20.0),
    )
    runoff = hydrologic_model(precip, pet, temp, params)
    assert runoff.shape == precip.shape
