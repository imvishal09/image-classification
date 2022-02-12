from email.mime import image
from pyexpat import model

# flask
from flask import Flask, render_template, request

# tensorflow and tf.keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16

# declare flask app
app = Flask(__name__)

# calling pre-trained model
model = VGG16()

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict():
    #saving input images in a folder
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # load image, resize image, make into an array and reshape
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat =  model.predict(image)
    label = decode_predictions(yhat)

    # return most likely result (highest probability)
    label = label[0][0]

    # classification
    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template('index.html', prediction=classification )


if __name__ == '__main__':
    app.run(port=5000, debug=True)

