# Reinforcement Learning for Portfolio Management

# Run the model on the local server
1. python ./rltensor/app/app_run.py
* Start the Flask application running at "http://127.0.0.1:5000"

2. Open the page with http://127.0.0.1:5000/fit/<start>/<end>/<load_file_path>/<int:num_epochs>
* start, end has the format: %yyyy-%mm-%dd %HH:%MM:%SS, e.g. 2015-01-01: 00:00:00.
* load_file_path is either "none" or the filepath to the file to load.
* num_epoch has to be positive integer

3. from  rltensor.app import APPAgent and then run APPAgent.udpate_data(), APPAgent.update_model(), and APPAgent.predict() everytime you store new data to sql database. 
