#-*- coding: utf-8 -*-

import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from flask_pymongo import Pymongo
from werkzeug.utils import secure_filename

# UPLOAD_dossier = 'uploads/'
#dossier destination des téléchargements
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'svg'])
#les extensions autorisés à être télécharger (ici on n'accepte que des images de type png, jpg, jpeg)

app = Flask(__name__)
app.config['MONGO_URI']= 'mongodb://192.168.150.110:27017'
mongo = Pymongo(app)

def fichier_autorise(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def telechargeFichier():
        if 'file' not in request.files:
            flash('No File Part')
            print('2')
            return redirect(request.url)
        file= request.files['file']
        if file.filename=='':
            flash('no selected file')
            return redirect(request.url)
        if file and fichier_autorise(file.filename):
            mongo.save_ffile(file.filename, file)
            mongo.db.users.insert({'username': 'root', 'image': file.filename})

            return 'ok'
            # filename= secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_dossier'], filename))

            # return os.popen(f"python3 predict_it.py 'uploads/{filename}'").read()
         

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
