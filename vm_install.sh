#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

#######################################
# System & Dependencies
#######################################
sudo apt-get update -y

#######################################
# Install SUMO (Traffic Simulation)
#######################################
sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get install -y sumo sumo-tools sumo-doc
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
export SUMO_HOME="/usr/share/sumo"
export LIBSUMO_AS_TRACI=1

#######################################
# Git Clone Repository
#######################################
git clone -b kelvin/training --single-branch https://github.com/EEEEEclectic/traffic_signal_control.git
cd traffic_signal_control

#######################################
# Conda & Python Environment
#######################################
# Create and activate the conda environment
conda create -n sumo_rl python=3.10 pandas numpy matplotlib seaborn -y
conda init
conda activate sumo_rl

# Upgrade pip inside the conda environment
pip install --upgrade pip

#######################################
# Install Project & Dependencies
#######################################
# Install project dependencies
pip install -e .

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install PyTorch Geometric
#TODO: Start from here
conda install pyg -c pyg -y

# Install tensorboard for logging
pip install tensorboard

#######################################
# Final Confirmation
#######################################
echo "Installation completed successfully!"
