# HydroPy

HydroPy is a minimal collection of hydrologic utilities built around
[JAX](https://github.com/google/jax). It provides a tiny example of a
process-based hydrologic model with modular snow, canopy, soil and
groundwater components.

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

precip = jnp.array([1.0, 0.5, 0.0])
temp = jnp.array([2.0, -1.0, 3.0])
params = HydroParams(
    snow=SnowParams(melt_factor=0.1),
    canopy=CanopyParams(max_storage=0.2),
    soil=SoilParams(infiltration_rate=0.3, field_capacity=0.5),
    groundwater=GroundwaterParams(baseflow_coeff=0.5),
)

runoff = hydrologic_model(precip, temp, params)
print(runoff)
```

## Development

Run the tests with:

```bash
pytest -q
```
