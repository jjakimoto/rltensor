# Reinforcement Learning for Portfolio Management

# Run the EIIE agent on the local server
1. python ./rltensor/app/app_run.py
* Start the Flask application running at "http://127.0.0.1:5000"

2. Use APPAgent.fit under the file, rltenor/app/app_agent.py.
* start, end has to follow the format: %yyyy-%mm-%dd %HH:%MM:%SS, e.g. 2015-01-01: 00:00:00.
* If end is not feeded, the most recent date will be used.
* load_file_path is either "none" or the filepath to the file to load.
* num_epoch has to be positive integer

3. Every time you update SQL data base, execute APPAgent.udpate_data(), which takes in the newest data into the model
4. Update model with APPAgent.update_model()
5. APPAgent.predict().

Once you have trained the model, just iterate the processes 3-5.

You can see an example notebook, examples/APPAgent.ipynb.
