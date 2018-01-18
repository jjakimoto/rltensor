from .core import Event


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """

    def __init__(self, strategy_id, symbol, datetime, signal_type, strength=1.0, value=None):
        """
        Initialises the SignalEvent.

        Args:
            strategy_id: int, The unique identifier for the strategy that
                generated the signal.
            symbol: str, The ticker symbol, e.g. ’GOOG’.
            datetime: datetime.datetime, The timestamp at which the signal was generated.
            signal_type: str, ’LONG’ or ’SHORT’.
            strength: float, An adjustment factor "suggestion" used to scale
                quantity at the portfolio level. Useful for pairs strategies.
        """
        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength
        self.value = value
