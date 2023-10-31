import carla_tester_model as CTmodel
import carla_tester_view as CTview
import carla_tester_presenter as CTpresenter

# Main application logic
if __name__ == "__main__":
    root = tk.Tk()
    model = CarlaWeatherModel()
    view = CarlaWeatherView(root)
    presenter = CarlaWeatherPresenter(model, view)
    
    view.create_town_selector()
    view.create_sliders()
    view.create_randomization_controls()
    
    presenter.initialize()
    view.run()