import json
import datetime

class ExperimentLogger:
    def __init__(self, experiment_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{experiment_name}_{timestamp}.json"
        self.experiment_name = experiment_name
        self.log_data = {'experiment_name': experiment_name, 'iterations': []}
        self.current_iteration = None


    def start_iteration(self, iteration_number, iteration_info):
        # Customize the iteration information here
        self.current_iteration = {'iteration_number' : iteration_number , 'iteration_info': iteration_info, 'frames': []}

    def save_frame(self, frame_dict):
        if self.current_iteration is not None:
            self.current_iteration['frames'].append(frame_dict)

    def end_iteration(self, end_iteration_info):
        if self.current_iteration is not None:
            # Add end-of-iteration information
            self.current_iteration['iteration_metrics'] = end_iteration_info
            self.log_data['iterations'].append(self.current_iteration)
            self.current_iteration = None
            self._save_to_file()

    def _save_to_file(self):
        with open(self.filename, 'w') as json_file:
            json.dump(self.log_data, json_file, indent=2)

    def finish(self):
        # Save any remaining data before finishing
        self._save_to_file()
        self.log_data = None