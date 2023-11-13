import tkinter as tk
import random
from PIL import Image, ImageTk
import numpy as np
import cv2

class CamerasView(tk.Tk):
    def __init__(self, root):
        self.root = root
        self.root.geometry("1200x600")

        self.left_frame = tk.Frame(self.root, bg="white")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

        self.create_buttons()

        # Save canvas objects as instance variables
        self.rgb_canvas = None
        self.depth_canvas = None
        self.segmentation_canvas = None
        self.bboxes_canvas = None
        self.create_canvas_widgets()

        #self.protocol("WM_DELETE_WINDOW", self.on_close)


    def on_close(self):
        # Handle any cleanup operations or confirm exit
        # This might involve closing connections, stopping processes, etc.
        self.destroy()  # Close the application window

    def create_buttons(self):
        button_01 = tk.Button(self.left_frame, text=f"Init", command=self.on_button_01_click)
        button_01.grid(row=0, column=0, sticky="ew")
        button_next = tk.Button(self.left_frame, text=f"Next", command=self.on_button_next_click)
        button_next.grid(row=1, column=0, sticky="ew")
        button_next = tk.Button(self.left_frame, text=f"Previous", command=self.on_button_next_click)
        button_next.grid(row=2, column=0, sticky="ew")


    def set_presenter(self, presenter):
        self.presenter = presenter


    def on_button_01_click(self):
        self.presenter.on_button_01_click()
        pass


    def on_button_next_click(self):
        self.presenter.on_button_next_click()


    def on_button_previous_click(self):
        self.presenter.on_button_previous_click()


    def create_canvas_widgets(self):
        # Create 4 canvas widgets with random colors
        colors = ["black"]

        for i in range(2):
            for j in range(2):
                canvas = tk.Canvas(self.right_frame, bg=random.choice(colors))
                canvas.grid(row=i, column=j, sticky="nsew")

                # Save the created canvas widgets to respective instance variables
                if i == 0 and j == 0:
                    self.rgb_canvas = canvas
                elif i == 0 and j == 1:
                    self.depth_canvas = canvas
                elif i == 1 and j == 0:
                    self.segmentation_canvas = canvas
                elif i == 1 and j == 1:
                    self.bboxes_canvas = canvas


    def update_canvas_views(self, sensor_data):
        if 'rgb' in sensor_data:
            self.update_rgb_canvas(sensor_data['rgb']['image'])
        if 'depth' in sensor_data:
            self.update_depth_canvas(sensor_data['depth']['image'])
        #if 'segmentation' in sensor_data:      
        #    self.update_segmentation_canvas(sensor_data['segmentation']['image'])
        #if 'map' in sensor_data:      
        #    self.update_map_canvas(sensor_data['map']['image'])
        if 'instance' in sensor_data:      
            self.update_instance_canvas(sensor_data['instance']['image'])
        if 'bboxes' in sensor_data:      
            self.update_bboxes_canvas(sensor_data['bboxes']['image'])
        if 'preds' in sensor_data:      
            self.update_preds_canvas(sensor_data['preds']['image'])
        if 'instseg_img' in sensor_data:      
            self.update_instance_canvas(sensor_data['instseg_img']['image'])




    def map_labels_to_colors(self, image_data):
        # Replace segmentation labels with distinct colors
        unique_labels = np.unique(image_data[:, :, 2])  
        label_colors = {label: np.random.randint(0, 256, size=3) for label in unique_labels}
        # Create an empty array to store the colored segmentation image
        colored_image = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
        # Map labels to colors in the segmentation image
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                label = image_data[i, j, 2]  # Assuming segmentation labels are in the third channel
                colored_image[i, j] = label_colors[label]
        return colored_image


    def update_segmentation_canvas(self, image_data):
        if self.segmentation_canvas:
            image_data = self.map_labels_to_colors(image_data)
            pil_image = Image.fromarray(image_data)
            pil_image = pil_image.resize((self.rgb_canvas.winfo_width(), self.rgb_canvas.winfo_height()), Image.ANTIALIAS)
            # Convert the resized image to a Tkinter PhotoImage
            image_data = ImageTk.PhotoImage(pil_image)
            # Update the canvas with the resized image
            self.segmentation_canvas.image = image_data  # Keep a reference to the image to prevent garbage collection
            self.segmentation_canvas.create_image(0, 0, anchor="nw", image=image_data)


    def update_rgb_canvas(self, image_data):
        if self.rgb_canvas:
            # Assuming 'image_data' is a numpy array
            rgb_image = np.array(image_data)
            image_data = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(image_data)
            pil_image = pil_image.resize((self.rgb_canvas.winfo_width(), self.rgb_canvas.winfo_height()))#, Image.ANTIALIAS)
            # Convert the resized image to a Tkinter PhotoImage
            image_data = ImageTk.PhotoImage(pil_image)
            # Update the canvas with the resized image
            self.rgb_canvas.image = image_data  # Keep a reference to the image to prevent garbage collection
            self.rgb_canvas.create_image(0, 0, anchor="nw", image=image_data)


    def update_depth_canvas(self, image_data):
        if self.depth_canvas:
            # Assuming depth_image is your provided depth image
            depth_image = np.array(image_data)  # Replace this with your provided data
            # Convert the depth image to grayscale
            image_data = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
            pil_image = Image.fromarray(image_data)
            pil_image = pil_image.resize((self.rgb_canvas.winfo_width(), self.rgb_canvas.winfo_height()))#, Image.ANTIALIAS)
            # Convert the resized image to a Tkinter PhotoImage
            image_data = ImageTk.PhotoImage(pil_image)
            # Update the canvas with the resized image
            self.depth_canvas.image = image_data  # Keep a reference to the image to prevent garbage collection
            self.depth_canvas.create_image(0, 0, anchor="nw", image=image_data)


    def update_bboxes_canvas(self, image_data):
        if self.depth_canvas:
            bboxes_image = np.array(image_data)
            image_data = cv2.cvtColor(bboxes_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_data)
            pil_image = pil_image.resize((self.rgb_canvas.winfo_width(), self.rgb_canvas.winfo_height()), Image.ANTIALIAS)
            # Convert the resized image to a Tkinter PhotoImage
            image_data = ImageTk.PhotoImage(pil_image)
            # Update the canvas with the resized image
            self.bboxes_canvas.image = image_data  # Keep a reference to the image to prevent garbage collection
            self.bboxes_canvas.create_image(0, 0, anchor="nw", image=image_data)


    def update_map_canvas(self, image_data):
        if self.depth_canvas:
            pil_image = Image.fromarray(image_data)
            pil_image = pil_image.resize((self.rgb_canvas.winfo_width(), self.rgb_canvas.winfo_height()), Image.ANTIALIAS)
            # Convert the resized image to a Tkinter PhotoImage
            image_data = ImageTk.PhotoImage(pil_image)
            # Update the canvas with the resized image
            self.bboxes_canvas.image = image_data  # Keep a reference to the image to prevent garbage collection
            self.bboxes_canvas.create_image(0, 0, anchor="nw", image=image_data)

    
    def update_instance_canvas(self, image_data):
        if self.depth_canvas:
            pil_image = Image.fromarray(image_data)
            pil_image = pil_image.resize((self.rgb_canvas.winfo_width(), self.rgb_canvas.winfo_height()), Image.ANTIALIAS)
            # Convert the resized image to a Tkinter PhotoImage
            image_data = ImageTk.PhotoImage(pil_image)
            # Update the canvas with the resized image
            self.segmentation_canvas.image = image_data  # Keep a reference to the image to prevent garbage collection
            self.segmentation_canvas.create_image(0, 0, anchor="nw", image=image_data)

    
    def update_preds_canvas(self, image_data):
        if self.depth_canvas:
            preds_image = np.array(image_data)
            image_data = cv2.cvtColor(preds_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_data)
            pil_image = pil_image.resize((self.rgb_canvas.winfo_width(), self.rgb_canvas.winfo_height()), Image.ANTIALIAS)
            # Convert the resized image to a Tkinter PhotoImage
            image_data = ImageTk.PhotoImage(pil_image)
            # Update the canvas with the resized image
            self.depth_canvas.image = image_data  # Keep a reference to the image to prevent garbage collection
            self.depth_canvas.create_image(0, 0, anchor="nw", image=image_data)
    