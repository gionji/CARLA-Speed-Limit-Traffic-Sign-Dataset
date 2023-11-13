import tkinter as tk

import carla_tester.model.carla_tester_model as CTmodel
import carla_tester.view.cameras_view as CTview
import carla_tester.presenter.carla_tester_presenter as CTpresenter
import carla_tester.utils as utils


# Main application logic
if __name__ == "__main__":
    root = tk.Tk()

    model     = CTmodel.CarlaTesterModel()
    view      = CTview.CamerasView(root)
    presenter = CTpresenter.CarlaTesterPresenter(model, view)
    
    view.set_presenter( presenter )

    model.connect_to_carla()

    model.run_experiment_init()
    
    root.mainloop()