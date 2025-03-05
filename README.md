# Description

The Python code in this repository implements four neural network based non-linear self-interference cancellation methods.

# Contents

This python code is split into the following files:

1. _NNCancellation.py_: This file loads the measured testbed data and performs non-linear cancellation using just two feedforward neural networks. It also plots the PSDs of the resulting signals as well as the cancellation performance.
2. _polynomialCancellation.py_: This file loads the measured testbed data and performs non-linear cancellation using the baseline polynomial model. It also plots the PSDs of the resulting signals as well as the cancellation performance.
3. _fullduplex.py_: This file contains all the helper functions that are required to do linear cancellation and non-linear cancellation using the baseline polynomial model and it is imported in both NNCancellation.py and polynomialCancellation.py.
4. _CVNN.py_: This file loads the measured testbed data and performs non-linear cancellation using the CVNN model. It also plots the PSDs of the resulting signals as well as the cancellation performance.
5. _CV-CLDNN.py_: This file loads the measured testbed data and performs non-linear cancellation using the CV-CLDNN model. It also plots the PSDs of the resulting signals as well as the cancellation performance.
6. _HCRDNN.py_: This file loads the measured testbed data and performs non-linear cancellation using the HCRDNN model. It also plots the PSDs of the resulting signals as well as the cancellation performance.
7. _transformer.py_ : This file loads the measured testbed data and performs non-linear cancellation using the transformer model. It also plots the PSDs of the resulting signals as well as the cancellation performance.
8. _fdTestbedData20MHz10dBm.mat_: This file contains the measured testbed data.
