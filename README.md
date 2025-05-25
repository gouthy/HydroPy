# HydroPy

HydroPy is a minimal collection of hydrologic utilities built around
[JAX](https://github.com/google/jax). It includes a small distributed
hydrologic model with snow, canopy, soil and groundwater processes.

## Installation

Install the package (and JAX) with pip:

```bash
pip install -e .
```

Depending on your platform you may need to install `jax` and `jaxlib`
manually. See the [JAX installation guide](https://github.com/google/jax#installation)
for details.

## Usage

```python
import jax.numpy as jnp
from hydropy import (
    SnowParams,
    CanopyParams,
    SoilParams,
    GroundwaterParams,
    HydroParams,
    hydrologic_model,
)

# time x n_cells arrays
precip = jnp.array([[1.0, 0.5], [0.3, 0.2], [0.0, 0.1]])
temp = jnp.array([[1.0, -1.0], [2.0, 0.5], [1.5, -0.5]])

params = HydroParams(
    snow=SnowParams(melt_temp=0.0, melt_rate=1.0),
    canopy=CanopyParams(capacity=1.0, evap_coeff=0.1),
    soil=SoilParams(field_capacity=1.0, percolation_coeff=0.1),
    groundwater=GroundwaterParams(baseflow_coeff=0.05),
)

runoff = hydrologic_model(precip, temp, params)
print(runoff)
```

## Development

Run the tests with:

```bash
pytest -q
```
