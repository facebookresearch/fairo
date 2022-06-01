import serial
from serial.serialutil import SerialException
from math import ceil
import numpy as np
import array
import time


class SpideyHand:
    def __init__(self, comport="/dev/ttyACM2", baud=115200):
        self.ser = serial.Serial(comport, baud)
        # connected = self.client.connectToDevice(device=comport)
        if not self.ser.isOpen():
            raise Exception(
                "Communication with gripper on serial port: %s and baud rate: %d not achieved"
                % (comport, baud)
            )

        self.init_success = True

    def getStatus(self):
        """Request the status from the gripper and return it in the Robotiq2FGripper_robot_input msg type."""

        # Check if read was successful
        # if status is None:
        #     return False

        return True

    def _true_pressure_bar(self):
        self.ser.flush()
        # time.sleep(0.1) #TODO: make this wait until serial data is available
        # if self.ser.inWaiting() < 1:
        # 	return None
        packet = self.ser.read_until(b"\n")
        data_string = packet.decode("utf")
        true_pressure_split = data_string.split()
        time_millis = int(true_pressure_split[0])  # milliseconds
        desired_pressure = int(true_pressure_split[1])  # percent
        true_pressure = int(true_pressure_split[2])  # percent
        return desired_pressure, true_pressure

    def grasp(self):  # full actuation
        self.ser.write("F".encode("utf-8"))
        # true_grasp_pressure = self._true_pressure_bar()
        print("\nfull grasp")
        # return true_grasp_pressure #user can save and/or print this "currentpressure = mySpideyHand.grasp" <<object, not class

    def goto(self, actuation):  # send over pyserial
        # time.sleep(0.1)
        # actuation
        if actuation == 0:
            self.ser.write("Z".encode("utf-8"))
            pressures = self._true_pressure_bar()
            print(pressures)
        elif actuation == 25:
            self.ser.write("Q".encode("utf-8"))
            pressures = self._true_pressure_bar()
        elif actuation == 50:
            self.ser.write("H".encode("utf-8"))
            pressures = self._true_pressure_bar()
        elif actuation == 75:
            self.ser.write("T".encode("utf-8"))
            pressures = self._true_pressure_bar()
        elif actuation == 100:
            self.ser.write("F".encode("utf-8"))
            pressures = self._true_pressure_bar()

    def stop(self):
        self.ser.close()

        # self._update_cmd() #TODO: make retime changes --> must change the teensy side too
