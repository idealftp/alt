#-*- coding: utf-8 -*-

import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename

# UPLOAD_dossier = 'uploads/'
#dossier destination des téléchargements
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'svg'])
#les extensions autorisés à être télécharger (ici on n'accepte que des images de type png, jpg, jpeg)

app = Flask(__name__)
app.config['MONGO_URI']= 'mongodb://192.168.150.110:27017/firstDb'
mongo = PyMongo(app)

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
            return  'ok'

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
