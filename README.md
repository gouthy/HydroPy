# HydroPy

HydroPy provides a minimal, process based hydrologic model implemented with
[JAX](https://github.com/google/jax). The model couples snow, canopy, soil and
groundwater processes and can be applied over many locations using `jax.vmap`.

## Installation

Install the package (and JAX) with pip:

```bash
pip install -e .
```

Depending on your platform you may need to install `jax` and `jaxlib` manually.
See the [JAX installation guide](https://github.com/google/jax#installation) for
details.

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

params = HydroParams(
    snow=SnowParams(melt_temp=0.0, melt_rate=0.1),
    canopy=CanopyParams(capacity=1.0),
    soil=SoilParams(capacity=2.0, conductivity=0.1),
    groundwater=GroundwaterParams(recession=0.05),
)

# Arrays with shape (time, n_locations)
precip = jnp.array([[1.0, 0.5], [0.2, 0.2], [0.0, 0.1]])
evap = jnp.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
temp = jnp.array([[1.0, -1.0], [2.0, 0.0], [0.0, 1.0]])

runoff = hydrologic_model(precip, evap, temp, params)
print(runoff)
```

## Development

Run the tests with:

```bash
pytest -q
```
