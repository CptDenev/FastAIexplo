import gradio as gr
import spaces
import torch
from fastai.vision.all import *

import pathlib
import platform

if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

def is_conciergerie(x): return x[0].isupper()

learn = load_learner('conciergerie_learner.pkl')

categories = ('Chapelle','Couloir des prisonniers','Exterieur', 'Salle des gardes', 'Salle des gens d Armes')

@spaces.GPU
def classify_image(img):
    learn.model.to('cuda') 
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(height=192, width=192)
label = gr.Label()
examples = ['Chapelle01.jpg', 'Chapelle02.jpg', 'CouloirPrisonnier.jpg', 'Exterieur.jpg', 'SalleGarde01.jpg', 'SalleGarde02.jpg', 'SalleGensArmes.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)