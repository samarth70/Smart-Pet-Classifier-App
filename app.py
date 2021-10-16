

# DOGS VS CATS DATASET PREDICTION

## LOADING MODULES


# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install tensorflow-addons
# !pip install gradio


import tensorflow_addons as tfa
import gradio as gr
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from google_drive_downloader import GoogleDriveDownloader as gdd
# from tensorflow.keras import *
# import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
# import time


def getData(flid,path,unzp=False):
  return gdd.download_file_from_google_drive(file_id=flid,
                                    dest_path=path,
                                    unzip=unzp)


"""##LOADING SAVED MODEL"""

model1='1TNF6uZBvcIfEUwzIR8t4L1kuImxb6PES'
model2='1cK1cIYdczAoEPkiNZUqx2r1UqF2idcay'
model3='1ldVcjryLk-YFfLRyNYdut5WeLLNxJ8ab'
model = model1 #@param ["model1", "model2","model3"] {type:"raw"}
PATH='./saved_model/best_model.h5'
getData(flid=model,path=PATH)

# For example images
# gdd.download_file_from_google_drive(file_id='1LdB6ZE9vxPi4HNN2emqJSoP0ig9DiG10',
#                                     dest_path='/content/examples.zip',
#                                     unzip=True)


model=load_model('/saved_model/best_model.h5')

labels=['Cat','Dog']
NUM_CLASSES=2
IMG_SIZE=224
# ex=[['/content/dogs-cat-examples/cat2.jpg'],  
#     ['/content/dogs-cat-examples/cat3.jpg'],
#     ['/content/dogs-cat-examples/dog2.jpeg'],
#     ['/content/dogs-cat-examples/dog.jpeg']]

"""
## RUNNING WEB UI"""

def classify_image(inp):
  inp = inp.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
  inp = tf.keras.applications.vgg16.preprocess_input(inp)
  prediction = model.predict(inp).flatten()
  return {labels[i]: float(prediction[i]) for i in range(NUM_CLASSES)}

image = gr.inputs.Image(shape=(IMG_SIZE, IMG_SIZE))
label = gr.outputs.Label(num_top_classes=2)

gr.Interface(fn=classify_image, inputs=image, outputs=label, title='Cats Vs Dogs',height=600, width=1200).launch()




