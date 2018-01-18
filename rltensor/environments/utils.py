from datetime import datetime
from copy import deepcopy
import time
import os
import pandas as pd
import numpy as np


def calculate_pv_after_commission(w1, w0, commission_rate):
    """
    @:param w1: target portfolio vector, first element is btc
    @:param w0: rebalanced last period portfolio vector, first element is btc
    @:param commission_rate: rate of commission fee, proportional to the transaction cost
    """
    w1 = deepcopy(w1)
    w0 = deepcopy(w0)
    mu0 = 1
    mu1 = 1 - 2 * commission_rate + commission_rate ** 2
    while abs(mu1 - mu0) > 1e-10:
        mu0 = mu1
        mu1 = (1 - commission_rate * w0[0] -
            (2 * commission_rate - commission_rate ** 2) *
            np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / \
            (1 - commission_rate * w1[0])
    return mu1


def date2seconds(str_time):
    datetime_obj = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
    seconds = time.mktime(datetime_obj.timetuple())
    return seconds


def seconds2date(seconds):
    date_obj = time.localtime(seconds)
    str_time = '%04d-%02d-%02d %02d:%02d:%02d'
    return str_time % (date_obj.tm_year, date_obj.tm_mon, date_obj.tm_mday,
                       date_obj.tm_hour, date_obj.tm_min, date_obj.tm_sec)


def get_data(pair, data_dir, start, end, period=1800):
    datafile = os.path.join(data_dir, pair + ".csv")
    timefile = os.path.join(data_dir, pair)

    start_sec = date2seconds(start)
    end_sec = date2seconds(end)

    FETCH_URL = "https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%d&end=%d&period=%d"
    COLUMNS = ["date", "high", "low", "open",
               "close", "volume", "quoteVolume", "weightedAverage"]

    url = FETCH_URL % (pair, start_sec, end_sec, period)
    print("Get %s from %d to %d with period %d" % (pair, start_sec, end_sec, period))

    df = pd.read_json(url, convert_dates=False)

    if df["date"].iloc[-1] == 0:
        print("No data.")
        return

    end_time = df["date"].iloc[-1]
    ft = open(timefile, "w")
    ft.write("%d\n" % end_time)
    ft.close()
    outf = open(datafile, "w")
    df.to_csv(outf, index=False, columns=COLUMNS)
    outf.close()
    print("Finish.")
    time.sleep(5)


def convert_time(t):
    m = {
        'Jan': "01",
        'Feb': "02",
        'Mar': "03",
        'Apr': "04",
        'May': "05",
        'Jun': "06",
        'June': "06",
        'Jul': "07",
        'Aug': "08",
        'Sep': "09",
        'Oct': "10",
        'Nov': "11",
        'Dec': "12"
    }
    t_list = t.replace(",", "").split()
    t_list[0] = m[t_list[0]]
    return "-".join([t_list[2], t_list[0], t_list[1]])
