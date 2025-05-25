# HydroPy

HydroPy is a minimal collection of hydrologic utilities built around
[JAX](https://github.com/google/jax). It contains a small set of
process-based components (snow, canopy, soil, and groundwater) and a helper
``hydrologic_model`` that ties them together.

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
temp = jnp.array([1.0, 2.0, 3.0])
evap = jnp.array([0.2, 0.2, 0.2])

params = HydroParams(
    snow=SnowParams(melt_temp=0.0),
    canopy=CanopyParams(capacity=0.1),
    soil=SoilParams(capacity=1.0, evap_coeff=1.0),
    groundwater=GroundwaterParams(outflow_coeff=0.5),
)

runoff = hydrologic_model(precip, temp, evap, params)
print(runoff)
```

## Development

Run the tests with:

```bash
pytest -q
```
