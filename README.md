# HydroPy

HydroPy is a small set of hydrologic utilities built around
[**JAX**](https://github.com/google/jax). It provides a distributed,
process‑based hydrologic model composed of simple snow, canopy, soil and
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

# Example for two locations and three time steps
precip = jnp.array([
    [1.0, 0.5],
    [0.2, 0.1],
    [0.0, 0.0],
])
temp = jnp.array([
    [1.0, -1.0],
    [2.0, 0.0],
    [1.5, 1.0],
])
evap = jnp.full_like(precip, 0.1)

params = HydroParams(
    snow=SnowParams(melt_temp=0.0, melt_rate=1.0),
    canopy=CanopyParams(capacity=1.0, drip_coeff=0.1, evaporation_coeff=1.0),
    soil=SoilParams(capacity=2.0, percolation_rate=0.5, evap_coeff=1.0),
    groundwater=GroundwaterParams(recession_coeff=0.3),
)

runoff = hydrologic_model(precip, temp, evap, params)
print(runoff)
```

The model operates on arrays with shape `(time, n_locations)` and returns
runoff with the same shape.


## Development

Run the tests with:

```bash
pytest -q
```
