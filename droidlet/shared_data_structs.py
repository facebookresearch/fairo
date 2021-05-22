import time
from droidlet.base_util import TICKS_PER_SEC


class Time:
    def __init__(self):
        self.init_time_raw = time.time()

    # converts from seconds to internal tick
    def round_time(self, t):
        return int(TICKS_PER_SEC * t)

    def get_time(self):
        return self.round_time(time.time() - self.init_time_raw)

    def get_world_hour(self):
        # returns a fraction of a day.  0 is sunrise, .5 is sunset, 1.0 is next day
        return (time.localtime()[3] - 8 + time.localtime()[4] / 60) / 24

    def add_tick(self, ticks=1):
        time.sleep(ticks / TICKS_PER_SEC)


class ErrorWithResponse(Exception):
    def __init__(self, chat):
        self.chat = chat


class NextDialogueStep(Exception):
    pass
