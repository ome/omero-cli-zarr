import time


def print_status(t0: int, t: int, count: int, total: int) -> None:
    """ Prints percent done and ETA.
        t0: start timestamp in seconds
        t: current timestamp in seconds
        count: number of tasks done
        total: total number of tasks
    """
    percent_done = float(count) * 100 / total
    dt = t - t0
    if dt > 0:
        rate = float(count) / (t - t0)
        eta_f = float(total - count) / rate
        eta = time.strftime("%H:%M:%S", time.gmtime(eta_f))
    else:
        eta = "NA"
    status = f"{percent_done:.2f}% done, ETA: {eta}"
    print(status, end="\r", flush=True)
