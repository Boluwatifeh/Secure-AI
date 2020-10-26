## Installation Guide 

This notebook was developed on [Jovian.ml](https://www.jovian.ml), a platform for sharing data science projects online. You can "run" this tutorial and experiment with the code examples in a couple of ways: *using free online resources* (recommended) or *on your own computer*.

### Option 1: Running using free online resources (1-click, recommended)

The easiest way to start executing this notebook is to click the "Run" button at the top of the jovian page [via this link](https://jovian.ai/tifeasypeasy/introducing-privacy-preserving-tool), and select "Run on Binder". This will run the notebook on [mybinder.org](https://mybinder.org), a free online service for running Jupyter notebooks. You can also select "Run on Colab" or "Run on Kaggle", but you'll need to create an account on [Google Colab](https://colab.research.google.com) or [Kaggle](https://kaggle.com) to use these platforms.

### Option 2: Running on your computer locally

You'll need to install Python and download this notebook on your computer to run it locally. I recommend using the [Conda](https://docs.conda.io/en/latest/) distribution of Python. Here's what you need to do to get started:

This include a complete guide on the installation processes for variety of operating systems. 

# Step 1 - Install anaconda/conda and add it to PATH
- For Windows users, [check this link](https://www.datacamp.com/community/tutorials/installing-anaconda-windows) to install and add Anaconda to PATH
- For MacOS users,  [check this link](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html) to install and add conda to PATH
- For Linux users, [check this link](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) to install and add conda to PATH
- For chromebook users, [check this link](https://boluwatife.hashnode.dev/how-to-install-and-run-a-jupyter-notebook-on-chromebooks) to install and add conda to PATH

After successful installation of conda, open up your command prompt and run the following command just to be sure everything is fine
```bash
conda --version
```
You should get a version number corresponding to the version of conda you installed. 

# Step 2 - Create an environment 
Run the following command to create an environment in conda
```bash
$ conda create -n my_env python=3.6
$ conda activate my_env 
```
After activating the environment, the next thing is to install jupyter notebook and syft which basically install every other packages required for the tutorial
```bash
$ conda install jupyter notebook
$ pip install syft
```
If you're facing some issues installing syft locally, kindly skip the `pip install syft` command and move on as there is a room for installing syft via the jupyter notebook.

# Step 3 - Clone this repo 
```bash
$ git clone https://github.com/Boluwatifeh/Secure-AI.git
```

# Step 4
cd into the tutorials folder and run the notebooks
```bash
$ cd tutorials
$ jupyter notebook
```
