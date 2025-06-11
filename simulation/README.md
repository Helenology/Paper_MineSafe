# Section 2.5 - A Simulation Study

- This folder contains the code used to reproduce the results in Section 2.5 ("A Simulation Study") of the main paper.

## 🛠️ Installation
- Before running the code, please execute the following command in the **parent directory** via the terminal: `sh install.sh`
    > Note: Ensure that the versions of `tensorflow_probability` and `tensorflow` are properly matched to avoid compatibility issues.


## 🧾 Table of Contents

|                        Folder / File                        |                    Information                     |
|:--------------------------------------------------:|:--------------------------------------------------:|
| [[Fig4]Simulation.ipynb](./[Fig4]Simulation.ipynb) | Jupyter notebook containing Python code to reproduce the results shown in Figure 4 of the main paper. |
|           [mean-540.npy](./mean-540.npy)           |                  	Numpy array representing the mean image of size $(p, q)$.                   |
|                      [results/](./results/)                       |       	Directory containing all the generated CSV files and plots.                                             |
|[plot_simulation.R](./plot_simulation.R)| 	R script to generate Figure 4 using the CSV results in the [results/](./results/) folder.|
|[simu_auxiliary.py](./simu_auxiliary.py)| 	Python module containing auxiliary functions used in the simulation.|