# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Obtained from https://github.com/Danfoa/robotiq_2finger_grippers/blob/master/robotiq_modbus_rtu/src/robotiq_modbus_rtu/comModbusRtu.py

# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Robotiq, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Robotiq, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2012, Robotiq, Inc.
# Revision $Id$
#
# Modifed from the orginal comModbusTcp by Kelsey Hawkins @ Georgia Tech


"""@package docstring
Module comModbusRtu: defines a class which communicates with Robotiq Grippers using the Modbus RTU protocol. 

The module depends on pymodbus (http://code.google.com/p/pymodbus/) for the Modbus RTU client.
"""

from pymodbus.client.sync import ModbusSerialClient
from pymodbus.exceptions import ModbusIOException
from math import ceil


class communication:
    def __init__(self):
        self.client = None

    def connectToDevice(self, device):
        """Connection to the client - the method takes the IP address (as a string, e.g. '192.168.1.11') as an argument."""
        self.client = ModbusSerialClient(
            method="rtu",
            port=device,
            stopbits=1,
            bytesize=8,
            baudrate=115200,
            timeout=0.2,
        )
        if not self.client.connect():
            print("Unable to connect to %s" % device)
            return False
        return True

    def disconnectFromDevice(self):
        """Close connection"""
        self.client.close()

    def sendCommand(self, data):
        """Send a command to the Gripper - the method takes a list of uint8 as an argument. The meaning of each variable depends on the Gripper model (see support.robotiq.com for more details)"""
        # make sure data has an even number of elements
        if len(data) % 2 == 1:
            data.append(0)

        # Initiate message as an empty list
        message = []

        # Fill message by combining two bytes in one register
        for i in range(0, len(data) // 2):
            message.append((data[2 * i] << 8) + data[2 * i + 1])

        # To do!: Implement try/except
        try:
            self.client.write_registers(0x03E8, message, unit=0x0009)
        except:
            print("Modbus write operation failure")
            return False
        return True

    def getStatus(self, numBytes):
        """Sends a request to read, wait for the response and returns the Gripper status. The method gets the number of bytes to read as an argument"""
        numRegs = int(ceil(numBytes / 2.0))

        # To do!: Implement try/except
        # Get status from the device
        try:
            response = self.client.read_holding_registers(0x07D0, numRegs, unit=0x0009)
        except Exception as e:
            print(e)
            return None

        # When reading failes, response is of type None
        if response is None:
            #   print("Failed to receive status")
            return None
        # Newer versions of pymodbus returns a ModbusIOException instead
        elif type(response) is ModbusIOException:
            return None

        # Instantiate output as an empty list
        output = []

        # Fill the output with the bytes in the appropriate order
        for i in range(0, numRegs):
            output.append((response.getRegister(i) & 0xFF00) >> 8)
            output.append(response.getRegister(i) & 0x00FF)

        # Output the result
        return output
