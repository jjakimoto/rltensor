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
