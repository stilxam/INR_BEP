# Running JAX on Windows Subsystem for Linux (WSL)
JAX with gpu support doesn't work directly on Windows, but can be made to work through the use of Windows Subsystem for Linux 2. The process however can be rather involved. Below is a plan that worked on my machine, but I can't guarantee that it will work on yours too. 

If this doesn't work, just using JAX with the cpu shouldn't be too big of a problem: you can just do the development and debugging work on your local machine on the cpu, and do all of the computationally heavy stuff like training on Snellius. 

**NB Installing stuff from the internet on WSL might not always work properly when using a VPN.**
## Install / enable WSL 2
To run JAX using a GPU on Windows, you'll have to make use of WSL 2. Depending on your windows version you can follow the steps in https://learn.microsoft.com/en-us/windows/wsl/install or in https://learn.microsoft.com/en-us/windows/wsl/install-manual 

You might also have to manually enable virtualization first.

## Nvidia driver
Update your nvidia driver **in windows (so not in WSL)** if needed.

## Add Ubuntu distro in WSL
In WSL, add an Ubuntu 24.04 distro by typing `wsl --install Ubuntu-24.04` into a terminal. For help with this, see:
https://learn.microsoft.com/en-us/windows/wsl/basic-commands

## Install miniconda
Alternatively you can use venv or something similar to manage your environments. Installing miniconda can be done by copying the following and pasting it in the terminal in which your linux distribution is running
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
After installing, initialize it by typing
```bash
~/miniconda3/bin/conda init bash
```
See https://docs.anaconda.com/miniconda/#quick-command-line-install

## Create a conda environment / venv virtual environment
E.g. by typing
```
conda create --name inr_edu_24 python=3.10 "numpy<2.0"
```
The reason that the numpy version should be less than 2.0 is that some packages like matplotlib might not yet be compattible with numpy 2.0.

Then activate the environment by typing
```
conda activate inr_edu_24
```

## Install JAX
Now there are two routes. The first one is the easy route: as detailed on https://jax.readthedocs.io/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier simply type 
```
pip install --upgrade "jax[cuda12]"
```
After pip is done installing JAX, run python (by typing `python` in the terminal) and run the following:
```python
import jax
key=jax.random.key(123)
```
If this results in a warning message that no GPU/TPU is found, and that JAX is falling back to cpu, this route has failed. Otherwise, you have JAX running on GPU on WSL.

### Hard route
So if that didn't work, get rid of your linux distro in wsl (`wsl --unregister Ubuntu-24.04`) and start over from the "Add Ubuntu distro in WSL" point. When you get to install jax, instead of the above, do the following:

Follow the instructions in https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2 to install the relevant CUDA Toolkit **in your WSL ubuntu distro**. 


Run the following commands one by one:
```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.4.0/local_installers/cudnn-local-repo-ubuntu2404-9.4.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.4.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.4.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn

sudo apt-get -y install cudnn-cuda-12
```
See https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local 

Then follow the instructions here: https://jax.readthedocs.io/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder

And check that everything worked e.g. by running python and trying to generate a random key.


# Other notes
You'll also want to install **Pytorch** to run some of the examples locally. This should probably be the **cpu only** version of Pytorch as you might otherwise get conflicts with the wrong cuda verion being used. 

To do this with anaconda, run
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

To do this with pip run
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

You'll also want to install the `common_jax_utils` by running
```
pip install git+https://github.com/SimonKoop/common_jax_utils.git
```

and the following packages as well:
* matplotlib
* jupyter
* fire
* wandb

# Setting things up on Linux
Same instructions as on WSL but without the hassle of the windows/WSL stuff.

# Setting things up on Mac
See https://jax.readthedocs.io/en/latest/installation.html#install-apple-gpu