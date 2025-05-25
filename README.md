# HydroPy

HydroPy is a minimal collection of hydrologic utilities built around
[JAX](https://github.com/google/jax). It started with a simple bucket
rainfall–runoff model but now also includes a more physically motivated
workflow made up of snow, canopy, soil and groundwater processes.

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
    HydroParams,
    SnowParams,
    CanopyParams,
    SoilParams,
    GroundwaterParams,
    hydrologic_model,
)

precip = jnp.array([1.0, 0.5, 0.0])
temp = jnp.array([1.0, -1.0, 2.0])

params = HydroParams(
    snow=SnowParams(melt_temp=0.0, melt_rate=1.0),
    canopy=CanopyParams(capacity=0.2),
    soil=SoilParams(field_capacity=1.0, percolation_rate=0.1),
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
