import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from rltensor.database.config import URL
from rltensor.database.sql_declarative import Price30M, Base
from rltensor.database.utils import seconds2datetime


DATA_DIR = "/home/tomoaki/work/Development/cryptocurrency/data"


def store(session, ticker, date, open, high, low, close,
          weightedAverage, volume, quoteVolume):
    obj = Price30M(ticker=ticker, date=date, open=open, high=high,
                   low=low, close=close, weightedAverage=weightedAverage,
                   volume=volume, quoteVolume=quoteVolume)
    session.add(obj)
    session.commit()


def store_csv(data_dir=DATA_DIR, currency_type='USD'):
    # Establish connection
    engine = create_engine(URL)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    filenames = os.listdir(data_dir)
    for name in filenames:
        if '.csv' in name and name.startswith(currency_type):
            filepath = os.path.join(data_dir, name)
            df = pd.read_csv(filepath)
            ticker = name.split('.')[0]
            df_val = df.values
            for val in tqdm(df_val):
                data = dict(session=session,
                            ticker=ticker)
                for i, col in enumerate(df.columns):
                    if col == 'date':
                        data[col] = seconds2datetime(val[i])
                    else:
                        data[col] = val[i]
                store(**data)
    session.close()


if __name__ == '__main__':
    store_csv()
