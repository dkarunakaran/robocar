#!/bin/bash

echo
echo Setting up ROS environment variables:
echo Run below command on tht terminal
echo

# IP address for the robot car
echo "export ROS_IP=192.168.1.10"

# IP address for the machine which does training and inference
echo "export ROS_MASTER_URI=http://192.168.1.4:11311"

echo "Setup the host if that is not set"
echo "sudo nano /etc/hosts"
echo "add: <ros master ip address without port> <hostname of ros master>"
echo "Eg: 192.168.1.4 dhanoop-virtualbox"
