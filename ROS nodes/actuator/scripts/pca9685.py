#!/usr/bin/env python3

from __future__ import division
import time

# Import the PCA9685 module.
import Adafruit_PCA9685

class PCA9685:
    pwm = None

    def __init__(self, freq=60):
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(freq)

    def set_pwm_value(self, channel=1, pulse=420):
        self.pwm.set_pwm(channel, 0, pulse)
