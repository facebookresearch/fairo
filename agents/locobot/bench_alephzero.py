import a0
import signal


def callback(pkt):
    print(f"Got {pkt.payload}")


s = a0.RemoteSubscriber(
    "100.95.90.42", "some/topic", a0.INIT_AWAIT_NEW, a0.ITER_NEXT, callback
)

signal.pause()
