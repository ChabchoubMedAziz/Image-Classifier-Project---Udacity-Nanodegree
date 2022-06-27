import sys
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
from tensorflow.keras.models import load_model
import json
batch_size = 32
image_size = 224


class_names = {}

def process_image(image): 
   
    image = tf.cast(image, tf.float32)
    image= tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    
    return image
    

def predict(image, model, top_k):
    #takes a path and opens an image
    image = Image.open(image_path)
    #creates an array of image
    image = np.asarray(image)
    #turns the immage from unit8 to float32
    image = tf.cast(image, tf.float32)
    #Resizes the image
    image = tf.image.resize(image, (image_size, image_size))
    #Tensor: rescaling images to be between 0 and 1
    image /= 255
    #remove single-dimensional entries from the shape of an array.
    image = image.numpy().squeeze()

    #add an extra dimension back
    image = np.expand_dims(image, axis = 0)


    ps = model.predict(image)[0] #ps is a list of lists, we have only one, we lelect that one
    probabilities = np.sort(ps)[-int(top_k):len(ps)] # short top probabilities
    prbabilities = probabilities.tolist() #create a list for the probabilities
    classes = np.argpartition(ps, -int(top_k))[-int(top_k):] # get names of int classes
    classes = classes.tolist() #create a list of int classes
    names = [class_names.get(str(i + 1)).capitalize() for i in (classes)] # get class names
    return probabilities, names
    


if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    print(args)
    
    print('image_path:', args.arg1)
    print('model:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    image_path = args.arg1
    
    model = load_model(args.arg2,custom_objects={'KerasLayer':hub.KerasLayer},compile = False)


   
    #model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer} )

    top_k = args.top_k
    if top_k is None: 
        top_k = 5

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
   
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    print(classes)