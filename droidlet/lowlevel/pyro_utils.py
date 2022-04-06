import Pyro4
import time


def safe_call(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Pyro4.errors.ConnectionClosedError as e:
        msg = "{} - {}".format(f._RemoteMethod__name, e)
        raise RuntimeError(msg)
    except Exception as e:
        print("Pyro traceback:")
        print("".join(Pyro4.util.getPyroTraceback()))
        raise e


def pyro_retry_loop(fn, retry_sleep=0.25):
    while True:
        try:
            return fn()
        except Pyro4.errors.PyroError:
            time.sleep(retry_sleep)
