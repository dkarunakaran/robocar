#!/bin/bash

source /home/dhanoop/catkin_ws/devel/setup.bash

export PATH=/home/dhanoop/miniconda3/bin:$PATH
source activate auto_pilot
export PYTHONPATH="/home/dhanoop/miniconda3/envs/auto_pilot/lib/python3.4/site-packages:$PYTHONPATH"

cd /home/dhanoop/miniconda3/envs/auto_pilot/lib/python3.4/site-packages
sudo ln -s /usr/local/lib/python3.4/site-packages/tensorflow tensorflow

cd /home/dhanoop/catkin_ws/src/auto_pilot/scripts
rosrun auto_pilot auto_pilot.py
