import zerorpc
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()
from mock_data import HelloData

s = zerorpc.Server(HelloData())
s.bind("tcp://0.0.0.0:4242")
s.run()
