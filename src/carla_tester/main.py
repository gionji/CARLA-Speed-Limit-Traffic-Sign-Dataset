import tkinter as tk

import carla_tester.model.carla_tester_model as CTmodel
import carla_tester.view.cameras_view as CTview
import carla_tester.presenter.carla_tester_presenter as CTpresenter
import carla_tester.utils as utils

from carla_tester.model.experiment import Experiment
from carla_tester.model.experiment_sync import ExperimentSync

import time

# Main application logic
if __name__ == "__main__":
    root = tk.Tk()

    #model     = CTmodel.CarlaTesterModel()
    #view      = CTview.CamerasView(root)
    #presenter = CTpresenter.CarlaTesterPresenter(model, view)
    
    #view.set_presenter( presenter )

    #model.run_complete_experiment()
    
    ## Run the interface
    #root.mainloop()

    experiment_genetic = ExperimentSync()
    experiment_genetic.init()
    experiment_genetic.run()