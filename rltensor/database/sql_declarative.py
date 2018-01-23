from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

from rltensor.database.config import URL


Base = declarative_base()


class PriceMixin(object):
    id = Column(Integer, primary_key=True)
    create_date = Column(DateTime, default=func.now())
    ticker = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    close = Column(Float, nullable=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    weightedAverage = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    quoteVolume = Column(Float, nullable=True)


class Price30M(Base, PriceMixin):
    __tablename__ = 'price30m'
    timeframe = Column(String(10), default='30M')


if __name__ == '__main__':
    engine = create_engine(URL)
    Base.metadata.create_all(engine)
