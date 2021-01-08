"""
Copyright (c) Facebook, Inc. and its affiliates.

A simple rate limiter supporting two types of rate limiting:

1. Limit rate by each unique ip address (ip mode)
2. Limit rate by all incoming requests (global mode)

reference: https://stackoverflow.com/questions/47275206
"""
import os
import redis
import time

from flask import jsonify
from flask_limiter.util import get_remote_address
from functools import update_wrapper

redis_url = os.environ["REDIS_URL"]
r = redis.from_url(redis_url)


def get_global_ip_addr():
    return "global"


def get_unique_ip_addr():
    return get_remote_address()


RATE_LIMIT_FUNCS = {"global": get_global_ip_addr, "ip": get_unique_ip_addr}


class RateLimit(object):
    def __init__(self, key, max_requests, seconds):
        self.reset = int(time.time()) + seconds
        self.key = "rl_{key}".format(key=key)
        self.max_requests = max_requests
        self.seconds = seconds
        cnt = r.get(self.key)
        self.current = int(cnt) if cnt else 0
        if not self.over_limit:
            r.incr(self.key)
            r.expireat(self.key, self.reset)
        cnt = r.get(self.key)
        self.current = int(cnt) if cnt else 0

    remaining = property(lambda x: x.max_requests - x.current)
    over_limit = property(lambda x: x.current > x.max_requests)


def over_limit(limit):
    response = {
        "result": "Max number of requests exceeded",
        "limit": "{max_requests} requests allowed every {seconds} seconds".format(
            max_requests=limit.max_requests, seconds=limit.seconds
        ),
        "status": False,
    }
    response = jsonify(response)
    response.status_code = 400
    return response


def ratelimit(max_requests, seconds, key_func, over_limit=over_limit):
    def decorator(f):
        def rate_limited(*args, **kwargs):
            key = RATE_LIMIT_FUNCS[key_func]()
            rlimit = RateLimit(key, max_requests, seconds)
            if over_limit is not None and rlimit.over_limit:
                return over_limit(rlimit)
            return f(*args, **kwargs)

        return update_wrapper(rate_limited, f)

    return decorator
