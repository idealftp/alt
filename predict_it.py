# sys.path.append('.\env\Lib\site-packages')
import sys
from keras.models import  Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

def whatis(file):
    file = str(file)
    model= VGG16()
    img= load_img(file, target_size=(224,224)) #charger l'image
    img=img_to_array(img) # convertir le tableau en numpy
    img=img.reshape((1, img.shape[0], img.shape[1], img.shape[2])) # creer la collection d'image
    img=preprocess_input(img) # pretraiter l'image
    y=model.predict(img)  # predire la classe de l'image
    return('top 3:', decode_predictions(y, top=3)[0])


if __name__ == '__main__':
    import sys
    print(whatis(sys.argv[1]))
