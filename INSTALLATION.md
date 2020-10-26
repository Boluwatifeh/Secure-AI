## Installation Guide 

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
