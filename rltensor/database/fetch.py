import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from rltensor.database.config import URL
from rltensor.database.sql_declarative import Price30M, Base
from rltensor.database.utils import date2datetime


DATA_DIR = "/home/tomoaki/work/Development/cryptocurrency/data"


def fetch(start, end, tickers):
    engine = create_engine(URL)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    start = date2datetime(start)
    end = date2datetime(end)
    base_query = session.query(Price30M).filter(Price30M.date >= start).filter(Price30M.date <= end)
    data = {}
    for ticker in tickers:
        x = base_query.filter(Price30M.ticker == ticker).all()
        timeidx = [x_i.date for x_i in x]
        dict_val = dict(
            open=[x_i.open for x_i in x],
            high=[x_i.high for x_i in x],
            low=[x_i.low for x_i in x],
            close=[x_i.close for x_i in x],
            weightedAverage=[x_i.weightedAverage for x_i in x],
            volume=[x_i.volume for x_i in x],
            quoteVolume=[x_i.quoteVolume for x_i in x])
        df = pd.DataFrame(dict_val, index=timeidx)
        df = df.loc[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        data[ticker] = df
    return data


if __name__ == '__main__':
    start = '2015-06-01 00:00:00'
    end = '2017-09-01 00:00:00'
    tickers = ['USDT_BCH', 'USDT_ZEC']
    data = fetch(start, end, tickers)
    print(data)
