import a0

#########################################

import time

prefix = a0.cfg(a0.env.topic(), "/foo/bar", str)

i = 0
p = a0.Publisher("proc/image.a0")
while True:
    a0.update_configs()
    msg = f"{prefix} {i}"
    print(msg)
    p.pub(msg)
    i += 1
    time.sleep(1)

#########################################

# import signal
# import sys

# p = a0.Publisher("proc/image.a0")

# def callback(pkt):
#     fbrp.set_trace()
#     print(pkt.payload.decode())
#     p.pub(pkt.payload)

# _sub = a0.Subscriber(
#     "camera/image.a0",
#     a0.INIT_AWAIT_NEW,
#     a0.ITER_NEXT,
#     callback)

# signal.pause()

#########################################

# import asyncio


# async def main():
#     p = a0.Publisher("proc/image.a0")
#     async for pkt in a0.aio_sub(
#         "camera/image.a0", a0.INIT_AWAIT_NEW, a0.ITER_NEXT
#     ):
#         print(pkt.payload.decode())
#         p.pub(pkt.payload)


# asyncio.run(main())
