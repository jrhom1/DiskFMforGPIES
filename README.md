# DiskFM for GPIES
A modified packaging of usable functions and tools from DiskFM (Mazoyer et al. 2020) (https://ui.adsabs.harvard.edu/abs/2020SPIE11447E..59M/abstract) specifically applied for GPI imaging data.

This repository contains relevant python scripts and functions for Hom et al. (2024) (Accepted for publication in MNRAS). This includes the backend disk model modified from anadisk_model (Millar-Blanchaer et al. 2016; https://ui.adsabs.harvard.edu/abs/2016AJ....152..128M/abstract) and the generic SPF utilized in Hom et al. (2024). For the HR 4796A SPF measured in Milli et al. (2017), please contact Julien Milli.

This repository is continually updated, and Jupyter notebooks will be added to guide users in a step-by-step process in the near future.

# Dependencies
numpy

matplotlib

astropy

yaml

numba

scipy

emcee

pyKLIP

schwimmbad

# Other Necessary Items
These scripts assume that you have a directory containing a GPI spectral mode dataset. There is functionality for polarimetric mode datasets, but they are not specifically supported in these scripts (but will be in the future). These scripts often rely on each other, so make sure you are referencing paths to scripts correctly. File and directory referencing will be improved for the future.

# Acknowledgement and Citation

If you use this repository in your own analysis, please cite:

Hom, J., Patience, J., Chen, C.C., et al., 2024, Accepted for Publication in MNRAS -- if using the generic SPF, the modified DiskFM framework, and/or the modified anadisk_model framework

Mazoyer, J., Arriaga, P., Hom, J., et al., 2020, Proceedings of the SPIE, 11447E, 59M -- if using DiskFM in general

Millar-Blanchaer, M.A., Wang, J.J., Kalas, P., et al., 2016, The Astronomical Journal, 152, 128M -- if using anadisk_model in general

If using the generic SPF as generated in Hom et al. (2024), please cite:

Hedman, M.M. & Stark, C.C., 2015, The Astrophysical Journal, 811, 67H

Throop, H.B., Porco, C.C., West, R.A., et al., 2004, Icarus, 172, 59T

Hanner, M.S. & Newburn, R.L., The Astronomical Journal, 97, 254H

Schleicher, D.G., Millis, R.L., Birch P.V., 1998, Icarus, 132, 397S

Moreno, F., Pozuelos, F., Aceituno, F., et al., 2012, The Astrophysical Journal, 752, 136M

Hui, M.T., 2013, Monthly Notices of the Royal Astronomical Society, 436, 1564H

