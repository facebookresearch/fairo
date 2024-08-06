import mrp
import time

mrp.import_msetup(".")

mrp.cmd.up("redis")
time.sleep(0.5)

mrp.cmd.up("set_foo", wait=True)
mrp.cmd.up("get_foo", attach=True, wait=True)

mrp.cmd.down("redis", wait=True)
