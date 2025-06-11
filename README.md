# A Brief Overview

This repository contains Python code for the paper "Doubly Smoothed Density Estimation with Application to Miners’ Unsafe Act Detection."


## 🛠 Installation

- Before running the code, please execute the following command in the **parent directory** via the terminal: `sh install.sh`
    > ⚠️ Note: Ensure that the versions of `tensorflow_probability` and `tensorflow` are properly matched to avoid compatibility issues.
    `TensorFlow Probability` has strict version requirements for specific `TensorFlow` releases. Please refer to the [official release compatibility table](https://github.com/tensorflow/probability/releases) to select the appropriate version of `TensorFlow Probability` or upgrate your `TensorFlow` installation accordingly.


## 📊 Part I. Simulation

This section provides reproducible code for Section 2.5 ("A Simulation Study") of the main paper.

|                        Folder / File                        |                    Information                     |
|:--------------------------------------------------:|:--------------------------------------------------:|
| [simulation/](./simulation) | Contains all simulation scripts and result files. |
|           [install.sh](./install.sh)           |                  		Shell script for setting up the required environment.                   |
|           [models/](./models/)           |                  		Includes Python code for the CD, DS, and GPA models.                   |
|           [utils.py](./utils.py)           |                  		Utility functions used across simulation scripts.                   |


- 🔔 Detailed instructions and explanations are available in the [SIMULATION README](./simulation/README.md). 
- 🔔🔔 The main reproducible code for Section 2.5 can be found in [./simulation/[Fig4]Simulation.ipynb](./simulation/[Fig4]Simulation.ipynb).


### Part II. Real Data


## Models (for Comparison)

### Part I. Model Information

| Models | Code Source | More Info |
| --- | ----------- | ----------- |
| [RPCA](https://dl.acm.org/doi/abs/10.1145/1970392.1970395) | Codes of the RPCA method are modified from the repository [dganguli/robust-pca](https://github.com/dganguli/robust-pca). **$\ell_1$ norm bug**: see [Error in L1-norm implementation #11](https://github.com/dganguli/robust-pca/issues/11) | **An interesting finding**: I found one typo from the original manuscript of [RPCA](https://dl.acm.org/doi/abs/10.1145/1970392.1970395), whose threshold used in line 3 of Algorithm 1 is $\mu$. This should (or may) be a typo, which should be substituted by $1/\mu$. |
| [CSPCA](https://ieeexplore.ieee.org/document/7372472) | <font color=#008000>Codes are written by me because the authors provided no relevant codes (although I tried to search on the Internet as well as email the authors).</font> | **Discoveries & Bugs**: 【1】 This work claimed to have used [*Fast computation of the L1-principal component of real-valued data*](https://ieeexplore.ieee.org/document/6855164)'s fast algorithm for solving the $\ell_1$-PCA problem in (9)-(11) on page 353. The problem is that the fast algorithm provided only one (max) PC, while the CSPCA needs $d(\geq 1)$ PCs. Thus, to avoid the above problem, I directly used the RPCA algorithm instead of using the fast algorithm to solve the aforementioned $\ell_1$-PCA problem. 【2】 No algorithm is proposed by this work. Not to mention how to implement the TV minimization in this work.
| [SSSR](https://ieeexplore.ieee.org/document/8485415) | <font color=#008000>Codes are written by me because the authors provided no relevant codes (although I tried to search on the Internet as well as email the authors).</font> | 【1】[SLIC](https://github.com/achanta/SLIC) algorithm for superpixels; 【2】Features including: [LBP](https://github.com/arsho/local_binary_patterns) (Local Binary Patterns); |
| [PI]() | Original codes in `C` language are provided by the author.| |

### Part II. Model Path

| Models | Path | Desciption |
| --- | ----------- | ----------- |
| [RPCA](https://dl.acm.org/doi/abs/10.1145/1970392.1970395) |[MineSafe-2024/models/robust-pca-master/r_pca.py](MineSafe-2024/models/robust-pca-master/r_pca.py)| Python to implement the robust PCA (RPCA) algorithm. |


