# from rltensor.environments import TradeEnv
from pytrade_env.runners import RLEnv
from rltensor.configs import eiie_config
from rltensor.agents import EIIE
import tensorflow as tf
import pickle


class Context:
    commission_rate = 2.5e-3
    price_keys = ['open', 'high', 'low']
    volume_keys = []
    initial_capital = 1.0


def train_model(start, end=None, load_file_path=None, save_file_path=None,
                num_epochs=int(2e6), agent_cls=EIIE,
                symbols_file="ticker1.pkl"):
    file = open(symbols_file, "rb")
    symbols = pickle.load(file)
    print("******************************symbols*********************")
    print(symbols)
    print("start", start, "end", end)
    context = Context()
    context.start = start
    context.end = end
    env = RLEnv(symbols, context)

    conf = dict(
        action_spec={"type": "float", "shape": env.action_dim},
        state_spec={"type": "float", "shape": (env.num_stocks, 3)}
    )

    default_config = eiie_config()
    conf.update(default_config)

    fit_config = dict(
        start=start,
        end=end,
        num_epochs=num_epochs,
        log_freq=1000,
    )

    if save_file_path is None:
        _start = start.split(" ")
        _start = "_".join(_start)
        if end is None:
            save_file_path = 'params{}/model.ckpt'.format(_start)
        else:
            save_file_path = 'params{}-{}/model.ckpt'.format(start, end)

    tf.reset_default_graph()
    agent = agent_cls(env=env, load_file_path=load_file_path, **conf)
    agent.fit(**fit_config, save_file_path=save_file_path)
    return agent
