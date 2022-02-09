import a0
import time

if __name__ == "__main__":
    s = a0.SubscriberSync("latest_robot_state", a0.INIT_MOST_RECENT)

    seconds = int(s.read().payload)

    time_diff = time.time() - curr_state.timestamp.seconds
    log.info(f"Last robot state retrieved within {time_diff}s.")

    num_seconds_stale = 10
    assert (
        time_diff < num_seconds_stale
    ), f"Robot state too stale by {time_diff}s, expected within {num_seconds_stale}s."
