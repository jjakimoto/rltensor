from __future__ import print_function

import datetime
import numpy as np
import pprint
try:
    import Queue as queue
except ImportError:
    import queue
import time

class Backtest(object):
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """
    def __init__(
            self, csv_dir, symbol_list, symbol_list_init, initial_capital,
            heartbeat, start_date, end_date, data_handler,
            execution_handler, portfolio, strategy, print_freq=100, strategy_context=None,
        ):
        """
        Initialises the backtest.
        
        Args:
            csv_dir: str, The hard root to the CSV data directory.
            symbol_list: list(str), The list of symbol strings.
            intial_capital: float, The starting capital for the portfolio.
            heartbeat: float, Backtest "heartbeat" in seconds
            start_date: datetime.datetime, The start datetime of the strategy.
            data_handler: class, Handles the market data feed.
            execution_handler: class, Handles the orders/fills for trades.
            portfolio: class, Keeps track of portfolio current
                and prior positions.
            strategy: class, Generates signals based on market data.
        """
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.symbol_list_init = symbol_list_init
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.end_date = end_date
        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.strategy_context = strategy_context
        self.events = queue.Queue()
        # The count number of each event instances
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self.print_freq = print_freq
        self._generate_trading_instances()
      
    def reset(self, index=None, is_detrend=False):
        # The count number of each event instances
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self.events = queue.Queue()
        self.data_handler.reset(self.events, index, is_detrend)
        # Initialize bar
        self.data_handler.update_bars()
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, 
                                            self.start_date, self.initial_capital)
        self.strategy.reset(self.portfolio, self.data_handler, self.events)
        self.execution_handler = self.execution_handler_cls(self.events)
    
    """
    def reset(self):
        self.__init__(self.csv_dir, self.symbol_list, self.symbol_list_init, self.initial_capital,
            self.heartbeat, self.start_date, self.end_date, self.data_handler_cls,
            self.execution_handler_cls, self.portfolio_cls, self.strategy_cls, self.print_freq, self.strategy_context)
    """
        
    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        self.data_handler = self.data_handler_cls(self.events, self.csv_dir, self.symbol_list, 
                                                  self.symbol_list_init, self.start_date, self.end_date)
        self.num_data = self.data_handler.num_data
        # Initialize bar
        self.data_handler.update_bars()
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, 
                                            self.start_date, self.initial_capital)
        self.strategy = self.strategy_cls(self.portfolio, self.data_handler, self.events, self.strategy_context)
        self.execution_handler = self.execution_handler_cls(self.events)
        
    def _run_backtest(self, is_display=True):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            if is_display and self.print_freq is not None:
                if i % self.print_freq == 0:
                    print("%d th" % (i+1))
            if self.data_handler.continue_backtest is False:
                break
            # Loop and handle all elements of self.events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            # Generate signal object to make orders
                            self.strategy.calculate_signals(event, i)
                            # Update current holdings, cash, and total assetsize
                            self.portfolio.update_timeindex(event)
                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            # Add OrderEvent to portfolio
                            self.portfolio.update_signal(event)
                        elif event.type == 'ORDER':
                            self.orders += 1
                            # Make fill event for backtest
                            self.execution_handler.execute_order(event)
                        elif event.type == 'FILL':
                            self.fills += 1
                            # Update holdings with fill event made by excute handler
                            self.portfolio.update_fill(event)
            # Update the market bars
            self.data_handler.update_bars()
            i += 1
            time.sleep(self.heartbeat)
            
    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        print("Creating summary stats...")
        stats = self.portfolio.output_summary_stats()
        print("Creating equity curve...")
        print(self.portfolio.equity_curve.tail(10))
        pprint.pprint(stats)
        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)
        
    def simulate_trading(self, is_detrend=False, is_display=True):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        if is_detrend:
            self.reset(is_detrend=is_detrend)
        self._run_backtest(is_display=is_display)
        self.portfolio.create_equity_curve_dataframe()
        if is_display:
            self._output_performance()
        result = self.get_result()
        return result, self.portfolio.equity_curve
        
    def get_result(self):
        stats = self.portfolio.get_stats()
        return stats