from matplotlib import pyplot as plt
from matplotlib import gridspec
import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

mp_selfie = mp.solutions.selfie_segmentation

def segment(image, color_choice): 
    with mp_selfie.SelfieSegmentation(model_selection=0) as model: 
        res = model.process(image)
        mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5 
        if color_choice == "Yellow":
            color = np.array([255, 222, 89])
        elif color_choice == "Light Grey":
            color = np.array([211, 211, 211])
        elif color_choice == "Light Blue":
            color = np.array([173, 216, 230])
        elif color_choice == "White":
            color = np.array([255,255,255])
        elif color_choice == "Black":
            color = np.array([0,0,0])
        elif color_choice == "Blur":
            blurred = cv2.blur(image, (40, 40))
            color = np.where(mask, image, blurred)
        else:
            color = np.array([255, 222, 89]) # default to yellow
        return np.where(mask, image, color)

color_choice = gr.inputs.Dropdown(["Yellow", "Light Grey", "Light Blue","White","Black","Blur"], label="Background Color")
webcam = gr.inputs.Image(shape=(640, 480), source="webcam")
webapp = gr.Interface(fn=segment, inputs=[webcam, color_choice], outputs="image", 
                      title="Virtual Background", description="Replace your background with a virtual one!", 
                      )
webapp.launch(inline=False)
