#-*- coding: utf-8 -*-

import os
from keras.models import  Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from flask_pymongo import PyMongo
# from werkzeug.utils import secure_filename

# UPLOAD_dossier = 'uploads/'
#dossier destination des téléchargements
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'svg'])
#les extensions autorisés à être télécharger (ici on n'accepte que des images de type png, jpg, jpeg)

app = Flask(__name__)
app.config['MONGO_URI']= 'mongodb://192.168.150.110:27017/firstDb'
mongo = PyMongo(app)


def whatis(file):
    file = str(file)
    model= VGG16()
    img= load_img(file, target_size=(224,224)) #charger l'image
    img=img_to_array(img) # convertir le tableau en numpy
    img=img.reshape((1, img.shape[0], img.shape[1], img.shape[2])) # creer la collection d'image
    img=preprocess_input(img) # pretraiter l'image
    y=model.predict(img)  # predire la classe de l'image
    return('top 3:', decode_predictions(y, top=3)[0])


def fichier_autorise(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def telechargeFichier():
    if request.method == "POST":
        file = request.files['file']
        if file and fichier_autorise(file.filename):
            print('ito e >>', file.filename)
            mongo.save_file(file.filename, file)
            mongo.db['files'].insert({'username': 'root', 'image': file.filename})
            return  whatis(mongo.send_file(file.filename))

    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form action="" method="post" enctype="multipart/form-data">
          <input type="file" name="file" />
          <input type="submit" value="Upload" />
        </form>
        '''


if __name__ == "__main__":
    app.secret_key = 'secret'
    app.debug = True
    app.run(host='0.0.0.0', port=80, threaded=False)
