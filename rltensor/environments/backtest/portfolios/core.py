from __future__ import print_function

import datetime
from math import floor
try:
    import Queue as queue
except ImportError:
    import queue
import numpy as np
import pandas as pd
from copy import deepcopy

from backtest.events import FillEvent, OrderEvent
from backtest.utils import create_sharpe_ratio, create_drawdowns


class Portfolio(object):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, 5-min, 30-min, 60 min or EOD.
    The positions DataFrame stores a time-index of the
    quantity of positions held.
    The holdings DataFrame stores the cash and total market
    holdings value of each symbol for a particular
    time-index, as well as the percentage change in
    portfolio total across bars.
    """

    def __init__(self, bars, events, start_date, initial_capital=100000.0, initial_weights=None, asset_size=0.):
        """
        Initialises the portfolio with bars and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).
        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """

        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.all_positions = self.construct_all_positions(initial_weights, asset_size)
        self.current_positions = self.initialize_positions(initial_weights, asset_size)
        self.all_holdings = self.construct_all_holdings(initial_weights, asset_size)
        self.current_holdings = self.construct_current_holdings(initial_weights, asset_size)

    def construct_all_positions(self, weights, asset_size):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = self.initialize_positions(weights, asset_size)
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self, weights, asset_size):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = self.initialize_positions(weights, asset_size)
        total_spend = 0
        if weights is not None:
            for weight in weights.values():
                total_spend += weight * asset_size
        d['datetime'] = self.start_date
        d['capital'] = self.initial_capital - total_spend
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self, weights, asset_size):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = self.initialize_positions(weights, asset_size)
        total_spend = 0
        if weights is not None:
            for weight in weights.values():
                total_spend += weight * asset_size
        d['capital'] = self.initial_capital - total_spend
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def initialize_positions(self, weights, asset_size):
        if weights is None:
            weights = {}
        weights = deepcopy(weights)
        positions = {}
        for key in self.symbol_list:
            if key not in weights:
                weight = 0.
            else:
                weight = weights[key]
            price = self.bars.get_latest_bar_value(key, "adj_close")
            positions[key] = asset_size * weight / price
        return positions

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).
        Makes use of a MarketEvent from the events queue.
        """
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])

        # Update positions
        # ================
        dp = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        dp['datetime'] = latest_datetime

        for s in self.symbol_list:
            dp[s] = self.current_positions[s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = {}
        dh['datetime'] = latest_datetime
        dh['capital'] = self.current_holdings['capital']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['capital']

        for s in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[s] * \
                self.bars.get_latest_bar_value(s, "adj_close")
            dh[s] = market_value
            dh['total'] += market_value

        # Append the current holdings
        self.all_holdings.append(dh)

    def update_positions_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to
        reflect the new position.
        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1
        # Update positions list with new quantities
        self.current_positions[fill.symbol] += fill_dir*fill.quantity

    def update_holdings_from_fill(self, fill):
        """
        Takes a Fill object and updates the holdings matrix to
        reflect the holdings value.

        Parameters:
        fill - The Fill object to update the holdings with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        fill_cost = self.bars.get_latest_bar_value(fill.symbol, "adj_close")
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['capital'] -= (cost + fill.commission)
        self.current_holdings['total'] -= (cost + fill.commission)

    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)

    def generate_order(self, signal, default_quantity=1.0):
        """
        Simply files an Order object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.

        Parameters:
        signal - The tuple containing Signal information.
        """
        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength

        if signal.value is None:
            mkt_quantity = default_quantity
        else:
            mkt_quantity = self.get_quantity(symbol, signal.value)
        cur_quantity = self.current_positions[symbol]
        order_type = 'MKT'
        if direction == 'BUY':
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        if direction == 'SELL':
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')
        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')
        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
        return order

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        if event.type == 'SIGNAL':
            # We should change here
            order_event = self.generate_order(event, default_quantity=1.0)
            self.events.put(order_event)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        returns = curve['total'].pct_change()
        returns.values[0] = np.zeros_like(returns.values[0])
        curve['returns'] = returns
        curve['equity_curve'] = (1.0+curve['returns']).cumprod()
        self.equity_curve = curve


    def output_summary_stats(self):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']
        sharpe_ratio = create_sharpe_ratio(returns, periods=252*60*6.5)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown
        stats = [("Total Return", "%0.2f%%" % \
            ((total_return - 1.0) * 100.0)),
            ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
            ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
            ("Drawdown Duration", "%d" % dd_duration)]
        self.equity_curve.to_csv('equity.csv')
        return stats

    def get_stats(self):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']
        sharpe_ratio = create_sharpe_ratio(returns, periods=252*60*6.5)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown
        stats = {"Total Return": (total_return - 1.0) * 100.0,
                 "Sharpe Ratio": sharpe_ratio,
                 "Max Drawdown": max_dd * 100.0,
                 "Drawdown Duration": dd_duration}
        return stats

    def get_quantity(self, symbol, value, val_type="adj_close"):
        """Transform value into #stock unit"""
        price = self.bars.get_latest_bar_value(symbol, val_type)
        return value / price

    @property
    def weights(self, val_type="adj_close"):
        portfolio_values = self.portfolio_values
        asset_size = np.sum(list(portfolio_values.values()))
        weights_ = {}
        for symbol in self.symbol_list:
            weights_[symbol] = portfolio_values[symbol] / asset_size
        return weights_

    @property
    def portfolio_values(self):
        portfolio_values = {}
        for symbol in self.symbol_list:
            price = self.bars.get_latest_bar_value(symbol, "adj_close")
            portfolio_values[symbol] = self.current_positions[symbol] * price
        return portfolio_values

    @property
    def asset_size(self):
        return np.sum(list(self.portfolio_values.values()))

    def set_positions(self, weights, asset_size):
        self.all_positions = self.construct_all_positions(weights, asset_size)
        self.current_positions = self.initialize_positions(weights, asset_size)
        self.all_holdings = self.construct_all_holdings(weights, asset_size)
        self.current_holdings = self.construct_current_holdings(weights, asset_size)
