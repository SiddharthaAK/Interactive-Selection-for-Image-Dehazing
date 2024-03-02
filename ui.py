import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torch
from DehazingModel import AODnet 
from torchvision.io import read_image

class ImageDehazerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Dehazing App")

        # Variables
        self.image_path = None
        self.bbox_start = None
        self.temp_rectangle = None

        # UI Elements
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.open_image)
        self.upload_button.pack()

        self.dehaze_button = tk.Button(root, text="Dehaze Selected Area", command=self.dehaze_selected_area)
        self.dehaze_button.pack()

        # Load AODnet model
        self.aodnet_model = self.load_aodnet_model()

        # Event bindings for drawing bounding box
        self.canvas.bind("<Button-1>", self.draw_bbox_start)
        self.canvas.bind("<B1-Motion>", self.draw_bbox_update)
        self.canvas.bind("<ButtonRelease-1>", self.draw_bbox_release)

    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            img = Image.open(self.image_path)
            img.thumbnail((400, 400))  # Resize image to fit display
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.img = img_tk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def draw_bbox_start(self, event):
        self.bbox_start = (event.x, event.y)
        if self.temp_rectangle:
            self.canvas.delete(self.temp_rectangle)
        self.temp_rectangle = self.canvas.create_rectangle(0, 0, 0, 0, outline='red')

    def draw_bbox_update(self, event):
        x0, y0 = self.bbox_start
        x1, y1 = event.x, event.y
        self.canvas.coords(self.temp_rectangle, x0, y0, x1, y1)

    def draw_bbox_release(self, event):
        x0, y0 = self.bbox_start
        x1, y1 = event.x, event.y

        # Store the final coordinates or perform additional actions
        self.bbox_coordinates = (x0, y0, x1, y1)

    def load_aodnet_model(model_path):
        # Instantiate an AODnet model
        model = AODnet()

        # Load the model weights from the provided path
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()  # Set the model to evaluation mode

        return model

    def dehaze_selected_area(self):
        if hasattr(self, 'bbox_coordinates'):
            x0, y0, x1, y1 = self.bbox_coordinates

            # Extract the selected area from the original image
            selected_area = self.original_image.crop((x0, y0, x1, y1))

            # Convert selected_area to the format suitable for your ML model
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
                # Add any other transformations needed for your model input
            ])
            selected_area_tensor = transform(selected_area).unsqueeze(0)

            # Pass selected_area_tensor through your ML model to get the dehazed output
            model = AODnet()  # Instantiate your AODnet model
            dehazed_output = model(selected_area_tensor)

            dehazed_img_pil = transforms.ToPILImage()(dehazed_output.squeeze(0).cpu())
            dehazed_img_tk = ImageTk.PhotoImage(dehazed_img_pil)
            self.dehazed_canvas.img = dehazed_img_tk
            self.dehazed_canvas.create_image(0, 0, anchor=tk.NW, image=dehazed_img_tk)

    def display_dehazed_region(self, dehazed_region):
        # Display the dehazed result
        dehazed_tk = ImageTk.PhotoImage(dehazed_region)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=dehazed_tk)
        self.canvas.img = dehazed_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDehazerApp(root)
    root.mainloop()
