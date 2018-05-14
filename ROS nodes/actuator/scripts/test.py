#!/usr/bin/env python

from pca9685 import PCA9685

pca9685 = PCA9685()
pca9685.set_pwm_value(1, pulse=430)
