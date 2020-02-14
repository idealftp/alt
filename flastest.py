#-*- coding: utf-8 -*-

import os
from pymongo import MongoClient
import gridfs
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
database = MongoClient('127.0.0.1:27017').firstDb
app.config['MONGO_URI']= 'mongodb://192.168.150.110:27017/firstDb'
mongo = PyMongo(app)

def pred(fln):
    fs = gridfs.GridFS(database)
    with open('tmp.png', 'wb') as file_tmp:
        file_tmp.write(fs.find_one({'filename':fln}).read())
    return os.popen(f"python3 predict_it.py").read()


def fichier_autorise(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def telechargeFichier():
    if request.method == "POST":
        file = request.files['file']
        if file and fichier_autorise(file.filename):
            # with open('test.save', 'wb') as f:
            #     f.write(mongo.send_file(file.filename).data)
            return pred(file.filename)

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
