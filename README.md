# Bayesian Physics-Informed Graph Neural Network

_CS6208 Class Project by Apivich H., Gregory Lau and Zhuanghua Liu_

---

## Running the code

Requirements for project are kept in `requirements.txt`.

The subfolder `data` contains all the dataset used by our experiments organised in an HGNN-friendly format.

To run the simulations, do
```
python run.py --exp=nbody-n4 --method=vi --add_noise=false
```
Change `exp` argument to the appropriate experiment data, and `method` to the correct Bayesian inference method. The available methods are `vi`, `dropout` and `none` (don't use and Bayesian methods). See the Python script for other arguments.

After performing the simulations, to generate the plots, do
```
python plot.py --exp=nbody-n4 --method=vi
```
Similarly, change `exp` and `method` arguments as appropriate.