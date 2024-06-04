# fnirs-sim

This repository contains the Python code to simulate an fNIRS recording session for a block design, taking Gervain et al. ([2012](
https://doi.org/10.1162/jocn_a_00157
))'s design as an example.

## Requirements

Python >= 3.12 (earlier versions may work as well), and Python packages NumpPy, Matplotlib, and SciPy. To install them execute this on your console/terminal:

```bash
python -m pip install numpy matplotlib scipy
```

## Example

```python
import numpy as np
import matplotlib.pyplot as plt
from src import HRF

sfreq = 10
time = np.arange(0, 20, 1/sfreq)
hbo, hbr = hrf(time, ratio=0.25)

plt.plot(hbo)
plt.plot(hbr)
```

See [this Jupyter notebook](docs/index.ipynb) for a complete walkthrough.