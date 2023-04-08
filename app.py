import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,Activation
from tensorflow.keras.models import Sequential,load_model
from flask import Flask,render_template,request,flash,redirect
from werkzeug.utils import secure_filename
import os

app=Flask(__name__)
app.config['IMAGE_UPLOADS']='C:\\Users\\druth\\OneDrive\\Desktop\\Projects\\Hurricane_Damage\\static\\Images'

model = load_model('model.h5')
labels=['Not Damaged','Damaged']
def newImage(path,x,y):
    imgs=[]
    img=Image.open(path)
    img=img.convert('L')
    img=img.resize(size=(x,y))
    img=np.array(img,dtype=np.float16)/255
    img=img.reshape(img.shape[0],img.shape[1],1)
    imgs.append(np.array(img))
    return np.array(imgs)


def predict(path):
    x_new=newImage(path,28,28)
    y_pred=model.predict(x_new)
    y_pred = np.where(y_pred > 0.5,0,1)
    return labels[y_pred[0][0]]


# pred=predict('-93.571101_30.992109000000003.jpeg')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect',methods=['POST',"GET"])
def detect():
    if request.method=="POST":
        image=request.files['file']
        image.save(os.path.join(app.config['IMAGE_UPLOADS'],secure_filename(image.filename)))
        pred=predict(os.path.join(app.config['IMAGE_UPLOADS'],secure_filename(image.filename)))
        data={
            'prediction':f'{pred}',
            'file':f"{secure_filename(image.filename)}"
        }
        return render_template('detect.html',data=data)

if __name__=='__main__':
    app.run()


