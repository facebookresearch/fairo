import Pyro4


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
