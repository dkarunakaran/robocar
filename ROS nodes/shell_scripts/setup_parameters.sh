#!/bin/bash


rosparam set /START_RECORDING false

#Steering constants
rosparam set /MAX_LEFT_ANGLE 1 #Joystic value
rosparam set /MIN_RIGHT_ANGLE -1 #Joystic value
rosparam set /MAX_STEERING_PULSE 540 #Actuator value
rosparam set /MIN_STEERING_PULSE 280 #Actuator value
rosparam set /STEERING_CHANNEL 1 #Actuator value

#Throttle constants
rosparam set /MIN_THROTTLE -1 #Joystic value
rosparam set /MAX_THROTTLE 1 #Joystic value
rosparam set /ZERO_PULSE 370 #Actuator value
rosparam set /MIN_THROTTLE_PULSE 220 #Actuator value
rosparam set /MAX_THROTTLE_PULSE 520 #Actuator value
rosparam set /THROTTLE_CHANNEL 0 #Actuator value
rosparam set /CONST_THROTTLE 0.22 #Joystic value
rosparam set /MAX_REVERSE_THROTTLE -0.20 #Joystic value
rosparam set /THROTTLE_AUTO_DRIVE 0.24 ##Joystic value
rosparam set /CONST_THROTTLE_STEP 0.001 #Joystic value

rosparam set /SHOW_DEBUG false

