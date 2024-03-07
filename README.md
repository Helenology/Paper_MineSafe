# A Brief Taste

This repository provides python codes for the working paper ``Detecting Miners’ Unsafe Acts Automatically by Density Estimation and Deep Learning".




## Models (for Comparison)

### Part I. Model Information

| Models | Code Source | More Info |
| --- | ----------- | ----------- |
| PCA | Directly use xxx | The very classical PCA method under the $\ell_2$-norm.|
| [RPCA](https://dl.acm.org/doi/abs/10.1145/1970392.1970395) | Codes of the RPCA method are modified from the repository [dganguli/robust-pca](https://github.com/dganguli/robust-pca). **$\ell_1$ norm bug**: see [Error in L1-norm implementation #11](https://github.com/dganguli/robust-pca/issues/11) | **An interesting finding**: I found one typo from the original manuscript of [RPCA](https://dl.acm.org/doi/abs/10.1145/1970392.1970395), whose threshold used in line 3 of Algorithm 1 is $\mu$. This should (or may) be a typo, which should be substitued by $1/\mu$. |
| [CSPCA]() | <font color=#008000>Codes are written by me because the authors provided no relevant codes (although I tried to search on the Internet as well as email the authors).</font> | **An interesting finding**: This work cited [???]()'s fast algorithm for $\ell_1$-norm calculation. The problem is that [???]() only provided a fast algorithm for the max PC. The CSPCA needs more than one max PC. |
| [SSSR](https://ieeexplore.ieee.org/document/8485415) | <font color=#008000>Codes are written by me because the authors provided no relevant codes (although I tried to search on the Internet as well as email the authors).</font> | |
| [PI]() | Original codes in `C` languange are provided by the author (sent to me in e-mail). I re-write the codes in `Python`.| |

### Part II. Model Path

| Models | Path | Desciption |
| --- | ----------- | ----------- |
| [RPCA](https://dl.acm.org/doi/abs/10.1145/1970392.1970395) |[MineSafe-2024/models/robust-pca-master/r_pca.py](MineSafe-2024/models/robust-pca-master/r_pca.py)| Python to implement the robust PCA (RPCA) algorithm. |


