# Setup (General)
First, we discuss the general setup for local machines.

## (1) Installing MuJoCo:
For all the commands, change `mujoco200_linux` to `mujoco200_macos` if installing MuJoCo on a Mac device.
```bash
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
rm mujoco200_linux.zip
mkdir ~/.mujoco
mv mujoco200_linux ~/.mujoco/mujoco200
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

```

## (2) Copy MuJoCo Key
Place the mjkey.txt (shared on Ed) into `~/.mujoco/` folder.

## (3) Install dependencies:

Create the virtual environment, and install the dependencies in the requirements.txt.
```bash
virtualenv --python=/usr/bin/python3.7 hw3_env
source hw3_env/bin/activate
pip install -r requirements.txt
```

Run `python test_installation.py` to ensure that `mujoco-py` was installed correctly.

# Troubleshooting
It is likely you will run into issues while installing on `mujoco-py`. Some of these commands are likely to help with installation issues on linux:
```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
sudo apt install libpython3.7-dev # replace with python version you are running
```

# Setup (Azure)
Consider using the Azure setup if local installation fails. Additional credits will be provided if needed.

Follow the Azure guide to setup the virtual machine. Use the same configuration as described. SSH into the virtual machine and copy the starter code. The machine will be in a conda environment by default. Deactivate by running `conda deactivate`. To check the current conda environment, `conda env list` should have an asterisk in front of the base environment, and NOT py38_default. Now, we can start the setup:

- Run Step (1) and (2)
- Run the following commands to prepare for Step (3):

```bash
# ensure deactivated conda environment
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa # press [ENTER] when prompted
sudo apt install python3.7
sudo apt install virtualenv
sudo apt install libpython3.7-dev
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```
- Run Step (3)

Run `python test_installation.py` to ensure that `mujoco-py` was installed correctly.
