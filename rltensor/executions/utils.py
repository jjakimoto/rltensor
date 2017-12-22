import numpy as np


def calc_stats(log_return, accumulated_pv, peak_pv):
    accumulated_pv *= np.exp(log_return)
    if peak_pv < accumulated_pv:
        peak_pv = accumulated_pv
    draw_down = (peak_pv - accumulated_pv) / peak_pv
    return_ = np.exp(log_return) - 1.
    return return_, accumulated_pv, peak_pv, draw_down


def calc_sharp_ratio(returns, bench_mark=0, eps=1e-6):
    var = np.var(returns)
    mean = np.mean(returns)
    return (mean - bench_mark) / (np.sqrt(var) + eps)
