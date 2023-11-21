


class CarlaTesterPresenter:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        #self.view.bind_close_event(self.on_app_close)
        # Wire up events (for example, button click)
        #self.view.on_button_01_click(self.on_button_01_click)



    def on_button_01_click(self):
        sensor_data = self.model.run_one_iteration()
        #self.view.update_canvas_views( sensor_data )  


    def on_button_next_click(self):
        sensor_data, _, _, _ = self.model.experiment.next()
        self.view.update_canvas_views( sensor_data )  


    def on_button_previous_click(self):
        sensor_data = self.model.experiment.previous()
        self.view.update_canvas_views( sensor_data )  
        

    def on_app_close(self):
        # Handle application closure in the Presenter
        self.model.cleanup()  # Perform necessary cleanup operations in the Model
