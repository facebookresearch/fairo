import a0
import signal


def callback(pkt):
    print(f"Got {pkt.payload}")


s = a0.RemoteSubscriber(
    "localhost", "some/topic", a0.INIT_AWAIT_NEW, a0.ITER_NEXT, callback
)

signal.pause()
