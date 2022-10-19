import time


class Spinner:
    """Sleeps the right amount of time to roughly maintain a specific frequency.

    Args:
        hz: frequency (times called / second)

    """

    def __init__(self, hz: float = 0.0):
        self.dt = 1.0 / hz if hz > 0.0 else 0.0

        # Initialize
        self.t_spin_target = time.time() + self.dt

    def spin(self):
        """Called each time in a loop to sleep a duration which maintains a specific frequency."""
        # No spinning if no time interval is specified
        if self.dt <= 0.0:
            return

        # Spin: sleep until time
        t_sleep = self.t_spin_target - time.time()
        if t_sleep > 0:
            time.sleep(t_sleep)
        else:
            # TODO: log warning without stuttering loop
            # log.info("Warning: Computation time exceeded designated loop time.")
            self.t_spin_target += -t_sleep  # prevent accumulating errors
        self.t_spin_target += self.dt
