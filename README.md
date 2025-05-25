# HydroPy

HydroPy is a minimal collection of hydrologic utilities built around
[JAX](https://github.com/google/jax). It contains a simple empirical
rainfall--runoff model as well as a lightweight process-based model
including snow, canopy, soil and groundwater components for distributed
simulations.

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
from hydropy import RRParams, rainfall_runoff

precip = jnp.array([1.0, 0.5, 0.0])
evap = jnp.array([0.2, 0.2, 0.2])
params = RRParams(capacity=2.0, evap_coeff=1.0)

runoff = rainfall_runoff(precip, evap, params)
print(runoff)
```

The process-based model operates on 2-D arrays representing time and
multiple grid cells:

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

# three timesteps for two cells
precip = jnp.ones((3, 2))
temp = jnp.zeros((3, 2))
evap = jnp.full((3, 2), 0.2)

params = HydroParams(
    snow=SnowParams(),
    canopy=CanopyParams(),
    soil=SoilParams(),
    groundwater=GroundwaterParams(),
)

runoff = hydrologic_model(precip, temp, evap, params)

print(runoff)
```

## Development

Run the tests with:

```bash
pytest -q
```
