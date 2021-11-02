import a0
import signal

p = a0.Publisher("from/bob")

def callback(pkt):
    p.pub(a0.Packet([(a0.DEP, pkt.id)], f"Got {pkt.payload}"))


s = a0.Subscriber("from/alice", a0.INIT_AWAIT_NEW, a0.ITER_NEXT, callback)

signal.pause()
