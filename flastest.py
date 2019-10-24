import os
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_dossier = 'E:/pred/simple-keras-rest-api/uploads/'
#dossier destination des téléchargements
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
#les extensions autorisés à être télécharger (ici on n'accepte que des images de type png, jpg, jpeg)

app= Flask(__name__)
app.config['UPLOAD_dossier']= UPLOAD_dossier

def fichier_autorise(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(file):
    model = VGG16()
    img = load_img(UPLOAD_dossier+file, target_size=(224, 224))  # charger l'image
    img = img_to_array(img)  # convertir le tableau en numpy
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # creer la collection d'image
    img = preprocess_input(img)  # pretraiter l'image
    y = model.predict(img)  # predire la classe de l'image
    print('top 7:', decode_predictions(y, top=7)[0])  # ici on affiche que les tops 3 des prédictions, on peut changer
    # le decode predictions de keras retourne 1Class name, 2 class description, 3 score

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_dossier'], filename)

@app.route('/', methods=['GET', 'POST'])
def telechargeFichier():
    if request.method == 'POST':
        #vérifier si il y a un fichier
        print('1')
        if 'file' not in request.files:
            flash('No File Part')
            print('2')
            return redirect(request.url)
        file= request.files['file']
        if file.filename=='':
            flash('no selected file')
            print('3')
            return redirect(request.url)
        if file and fichier_autorise(file.filename):
            filename= secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_dossier'], filename))
            #je sauvegarde le fichier à uploader dans un dossier indiqué dans le path en haut
            print('4')
            print(file)
            print(filename) # filename le nom du fichier téléchargé (ici on n'accepte que des images de type png, jpg, jpeg)
            predict(filename)
            return redirect(url_for('uploaded_file', filename=filename))
            # à la fin de l'upload , je redirige le client à l'image uploader
            #os.remove(filename) supprimer le fichier  télécharger après prediction

        # return "Vous avez envoyé : {message}".format(message=request.files['contenuFormu'])
        #on utilise request.form pour les types text, request.file pour les types fichiers d'un formulaire
        #ici le message c'est le nom du variable à retourn au client
        # le contenForm c'est l'id du fich que le client a choisi
        # pour input, type text pour les textes et type file pour les fichi

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
    app.run()