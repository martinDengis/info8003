# Setting Up Jupyter Notebooks Locally

This guide provides step-by-step instructions to install and configure a local environment for running Jupyter notebooks.

## 1. Install VS Code (Recommended)
- Download and install Visual Studio Code from [this link](https://code.visualstudio.com/).

## 2. Install Miniconda

You can find Miniconda installation instructions in the [official Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html#installing).

Once Miniconda is installed, create a virtual environment dedicated to your projects using the following command:
```
conda create --name my_environment python=3.10
```
You can activate the environment using the command:
```
conda activate my_environment
```
and deactivate it using:
```
conda deactivate
```

## 3. Open a Jupyter Notebook
- Open the directory where your notebook is located in VS Code.
- Install the Jupyter extension in VS Code by navigating to the Extensions tab (`Ctrl+Shift+X`), searching for "Jupyter," and clicking "Install."
- Ensure your virtual environment is selected as the Python interpreter by pressing `Ctrl+Shift+P`, typing "Python: Select Interpreter," and choosing `my_environment`.
- If you prefer running Jupyter from the terminal, install Jupyter Notebook within your environment:
  ```bash
  conda install -c conda-forge notebook
  ```
  Then start Jupyter Notebook with:
  ```bash
  jupyter notebook
  ```
- A browser window will open, allowing you to interact with notebooks.

## 4. Installing Additional Packages
If your notebook requires additional libraries, install them within your environment using:
```bash
pip install <library_name>
```
Replace `<library_name>` with the specific package you need.

## 5. Additional Resources
For more Python resources, refer to [Gilles Louppe’s Python tutorial](https://github.com/glouppe/info8002-deep-learning/blob/master/README.md).
