from flask import Flask
import json
from copy import deepcopy
import numpy as np
from tqdm import tqdm

from rltensor.app.utils import train_model

app = Flask(__name__)

context = dict()


# Have to run this function to get model
@app.route("/fit/<start>/<end>/<path:load_file_path>/<int:num_epochs>")
def fit(start, end, load_file_path, num_epochs):
    # Preprocess
    start = start.split("_")
    start = start[0] + " " + start[1]
    if end == "none":
        end = None
    else:
        end = end.split("_")
        end = end[0] + " " + end[1]
    if load_file_path.lower() == "none":
        load_file_path = None
    agent = train_model(start, end, load_file_path=load_file_path,
                        num_epochs=num_epochs)
    context["agent"] = agent
    return 'success'


@app.route('/predict')
def predict():
    agent = context["agent"]
    recent_state = agent.get_recent_state()
    recent_actions = agent.get_recent_actions()
    actions = agent.predict(recent_state, recent_actions)
    agent.current_actions = actions
    actions = actions.tolist()
    return json.dumps(actions)


@app.route('/update_model/<int:num_epochs>')
def update_model(num_epochs):
    agent = context["agent"]
    pbar = tqdm()
    while True:
        agent.env.data_handler.update_bars()
        if agent.env.data_handler.continue_trading is False:
            break
        pbar.update(1)
        # prediction
        recent_state = agent.get_recent_state()
        recent_actions = agent.get_recent_actions()
        current_actions = agent.predict(recent_state, recent_actions)

        current_bars = agent.env.data_handler.get_current_bars()
        prev_bars = agent.env.data_handler.get_prev_bars()
        returns = current_bars['price'][:, 0] / prev_bars['price'][:, 0] - 1.
        # observation = self._get_observation(current_bars)
        observation = deepcopy(current_bars)
        terminal = False
        if not hasattr(agent, "prev_actions") or agent.prev_actions is None:
            prev_actions = deepcopy(current_actions)
        else:
            prev_actions = deepcopy(agent.prev_actions)
        trade_amount = np.sum(np.abs(current_actions[1:] - prev_actions[1:]))
        reward = np.sum(returns * current_actions[1:])
        cost = 0
        agent.prev_actions = deepcopy(current_actions)
        info = {
            'reward': reward,
            'returns': returns,
            'cost': cost,
            'trade_amount': trade_amount,
        }
        agent.observe(observation, current_actions,
                      reward, terminal, info,
                      training=False, is_store=True)
        # Start training
        for epoch in range(num_epochs):
            # Update parameters
            response = agent.nonobserve_learning()
    return 'success'


@app.route('/update_data')
def update_data():
    agent = context["agent"]
    agent.env.data_handler.update_data()
    return 'success'


# start app
if (__name__ == "__main__"):
    app.run(port=5000)
