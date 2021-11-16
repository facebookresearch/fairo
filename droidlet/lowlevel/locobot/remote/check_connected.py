import sys
import os
import Pyro4
import Pyro4.errors

f = open(os.devnull, 'w')
sys.stdout = f
sys.stderr = f

with Pyro4.Proxy("PYRONAME:" + sys.argv[1] + "@0.0.0.0") as p:
    try:
        p._pyroBind()
        sys.exit(0)
    except Pyro4.errors.CommunicationError:
        sys.exit(1)
