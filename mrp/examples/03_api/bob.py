import a0
import signal

s = a0.Subscriber("some/topic", lambda pkt: print(f"Got {pkt.payload}"))

signal.pause()
