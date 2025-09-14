# Dataset Generators Collection

Unified implementations for all synthetic datasets used to benchmark Online / Batch Kernel SDR.

## Directory Structure

```
data_gens/
├── __init__.py                 # unified interface + mapping
├── highly_nonlinear.py         # highly nonlinear manifold + mixed signals
├── extreme_nonlinear_1.py      # extreme nonlinear variant 1 (60D)
├── extreme_nonlinear_2.py      # extreme nonlinear variant 2 (80D)
├── extreme_nonlinear_3.py      # extreme nonlinear variant 3 (100D)
├── piecewise_nonlinear.py      # piecewise / discontinuous
├── swiss_nonlinear.py          # swiss roll + extended embedding
├── better_xor_data.py          # improved multi-output XOR (101D input)
└── nuclear_friendly_xor_data.py # kernel-friendly XOR multi-output
```

## Generator Overview

| Name | Input Dim | Output Dim | Complexity | Notes |
|------|-----------|------------|------------|-------|
| highly_nonlinear | 10 | 5 | Medium | Spiral + radial + polynomial mixes |
| extreme1 | 60 | 5 | High | Rich param nonlinear transforms |
| extreme2 | 80 | 5 | Very High | Deeper composite structure |
| extreme3 | 100 | 5 | Very High | Torus-like + harmonics |
| piecewise | 50 | 5 | Medium | Discontinuities + piecewise logic |
| swiss | 80 | 5 | High | Swiss roll manifold embedding |
| better_xor | 101 | 5 | High | XOR count + smooth regressors + cross interactions |
| nuclear_xor | 208 | 5 | High | Low linear corr, strong nonlinear structure |

## Usage

Unified interface:

```python
from data_gens import get_generator
gen = get_generator('highly_nonlinear')
X, Y = gen(n_samples=1500, noise_level=0.15, random_state=42)

gen = get_generator('extreme1')
gen = get_generator('better_xor')
gen = get_generator('nuclear_xor')
gen = get_generator('extreme_nuclear')
```

Direct import:

```python
from data_gens.better_xor_data import generate_better_xor_data
from data_gens.swiss_nonlinear import generate_swiss_nonlinear
X, Y = generate_better_xor_data(n_samples=1000, noise_level=0.1)
X, Y = generate_swiss_nonlinear(n_samples=1500, random_state=42)
```

## Adding a New Generator

1. Create a new file in `data_gens/`
2. Add mapping in `__init__.py`
3. Keep a consistent callable signature: `gen(n_samples, noise_level=..., random_state=...)`
4. Document assumptions (scale, SNR, drift, etc.)

## Recent Updates

* Unified interface across all generators
* Extended XOR variants to multi-output (5D targets)
* Added kernel-friendly XOR with drift & SNR controls
* Added extreme variants for stress testing high dimensionality
* Ensured all paths and names remain stable for benchmarking scripts
