import a0
import signal

s = a0.RemoteSubscriber(
    "localhost", "some/topic", callback=lambda pkt: print(f"Got {pkt.payload}")
)

signal.pause()
