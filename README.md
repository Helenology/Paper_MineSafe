# A Brief Taste

Relevant codes and some illustrations for the working paper ``Detecting Miners’ Unsafe Acts Automatically by Density Estimation and Deep Learning".



# Codes

## Methods for Comparison

- [RPCA](https://dl.acm.org/doi/abs/10.1145/1970392.1970395). Codes of the RPCA method are modified from the repository [dganguli/robust-pca](https://github.com/dganguli/robust-pca), with some bugs fixed and more annotations added.
  - **$\ell_1$ norm bug**: see [Error in L1-norm implementation #11](https://github.com/dganguli/robust-pca/issues/11)
  - **An interesting finding**: I found one typo from the original manuscript of [RPCA](https://dl.acm.org/doi/abs/10.1145/1970392.1970395), whose threshold used in line 3 of Algorithm 1 is $\mu$. This should (or may) be a typo, which should be substitued by $1/\mu$.
- [SSSR](https://ieeexplore.ieee.org/document/8485415). Codes of the SSSR method are written by me because the authors provided no relevant codes (although I tried to search on the Internet as well as email the authors).

