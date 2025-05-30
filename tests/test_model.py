import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

jax = pytest.importorskip("jax")
jnp = jax.numpy

from hydropy import (
    SnowParams,
    CanopyParams,
    SoilParams,
    GroundwaterParams,
    LandUseParams,
    HydroParams,
    HydroState,
    build_cell_params,
    hydrologic_model,
)


def test_hydrologic_model_shape():
    precip = jnp.ones((3, 2)) * 0.5
    evap = jnp.ones((3, 2)) * 0.1
    temp = jnp.ones((3, 2))
    cell_params = HydroParams(
        snow=SnowParams(day_frac=1.0),
        landuse=LandUseParams(
            canopy=CanopyParams(f_bare=0.5, f_veg=0.5, LAI=1.0, cap0=1.0),
            soil=SoilParams(
                S_so_max=2.0,
                S_so_wilt=0.1,
                S_so_grmin=0.2,
                S_so_grmax=1.0,
                S_so_sg_min=0.1,
                S_so_sg_max=2.0,
                b=1.0,
                R_gr_min=0.01,
                R_gr_max=0.02,
                dt=1.0,
            ),
            groundwater=GroundwaterParams(),
            imperv_frac=0.0,
        ),
    )
    params = build_cell_params([cell_params, cell_params])
    runoff = hydrologic_model(precip, evap, temp, params)
    assert runoff.shape == precip.shape
