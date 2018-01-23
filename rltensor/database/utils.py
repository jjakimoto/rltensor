from datetime import datetime
import time
from time import mktime


def date2seconds(str_time):
    datetime_obj = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
    seconds = time.mktime(datetime_obj.timetuple())
    return seconds


def seconds2date(seconds):
    date_obj = time.localtime(seconds)
    str_time = '%04d-%02d-%02d %02d:%02d:%02d'
    return str_time % (date_obj.tm_year, date_obj.tm_mon, date_obj.tm_mday,
                       date_obj.tm_hour, date_obj.tm_min, date_obj.tm_sec)


def seconds2datetime(seconds):
    time_obj = time.localtime(seconds)
    datetime_obj = datetime.fromtimestamp(mktime(time_obj))
    return datetime_obj


def date2datetime(str_time):
    datetime_obj = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
    return datetime_obj
