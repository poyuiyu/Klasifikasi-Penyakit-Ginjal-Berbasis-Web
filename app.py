from cProfile import label
import json
from flask import Flask, redirect, url_for, render_template, request, Response, jsonify, send_file, flash
import numpy as np 
from util import base64_to_pil
from werkzeug.utils import secure_filename

import tensorflow as tf 
from tensorflow import keras 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from tensorflow.keras.models import load_model 
from keras.preprocessing import image
from keras.utils import get_file
import seaborn as sns
import matplotlib.pyplot as plt


app = Flask(__name__)

model = load_model('models/modelCNN.h5')



def model_predict(img, model):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = x.reshape(-1, 224, 224, 3)
    x = x.astype('float32')
    x = x / 225.0
    preds = model.predict(x)
    return preds

@app.route('/submit_train', methods=['POST'])
def submit_train():
    import zipfile
    import os
    import warnings
    import glob
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.preprocessing import image
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.utils import plot_model
    import seaborn as sns
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import base64
    
    if request.method == 'POST':
        submit = request.form['traindata']
        val_size = 0.20

        Train_datagen = ImageDataGenerator(
            rotation_range = 30,
            brightness_range = [0.2,1.0],
            shear_range = 0.3,
            zoom_range = 0.3,
            horizontal_flip = True,
            fill_mode = "nearest",
            rescale = 1./255,
            validation_split = val_size
        )

        Validation_datagen = ImageDataGenerator(
            rotation_range = 30,
            brightness_range = [0.2,1.0],
            shear_range = 0.3,
            zoom_range= 0.3,
            horizontal_flip = True,
            fill_mode = "nearest",
            rescale = 1./255,
            validation_split = val_size
        
        )

        img_width = 150
        img_height = 150

        Train_generator = Train_datagen.flow_from_directory(
            submit,
            target_size = (img_width,img_height),
            color_mode = "rgb",
            class_mode = "categorical",
            batch_size = 32,
            shuffle = True,
            subset = "training"
        )

        Validation_generator = Validation_datagen.flow_from_directory(
            submit,
            target_size = (img_width,img_height),
            color_mode = "rgb",
            class_mode = "categorical",
            batch_size = 32,
            shuffle = False,
            subset = "validation"

        )
        conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 
        conv_base.trainable = False    
        Model = Sequential()                                                                               
        Model.add(conv_base)
        Model.add(GlobalAveragePooling2D())
        Model.add(Flatten())
        Model.add(Dense(128, activation='relu'))
        Model.add(Dropout(0.5, seed=10))
        Model.add(Dense(64, activation='relu'))
        Model.add(Dropout(0.5, seed=10))
        Model.add(Dense(4, activation='softmax'))
        opt = Adam(lr=0.0003)                       
        Model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
      
        batch_size = 10

        history = Model.fit(Train_generator, 
                            epochs =  1, 
                            steps_per_epoch = 100//batch_size, 
                            validation_data = Validation_generator, 
                            verbose = 2, 
                            validation_steps = 20//batch_size,
                            )
        from matplotlib.figure import Figure                    
        fig = Figure()                                       
        Model.save('static/modelcnn.h5')
        acc = history.history['accuracy']
        acc2 = history.history['val_accuracy']
        loss = history.history['loss']
        loss2 = history.history['val_loss']
        epochs = range(len(acc))

        fig, (axs1, axs2) = plt.subplots(1, 2)
        axs1.plot(epochs, acc, 'r', label = 'Train Accuracy')
        axs2.plot(epochs, loss, 'r', label='Train Loss')
        canvas = FigureCanvas(fig)
        img = io.BytesIO()
        fig.savefig(img)
        img.seek(0)
        return send_file(img, mimetype='img/png')

        # fig, ax = plt.subplots(figsize=(4,4))
        # ax = sns.set_style(style=('darkgrid'))
        # # axis = fig.add_subplot(1, 1, 1)
        # ax.plot(epochs, acc, 'r', label='Train Accuracy')
        # canvas = FigureCanvas(fig)
        # plt.plot(epochs, acc, 'r', label='Train accuracy')
        # plt.plot(epochs, acc2, 'g', label='Val Train accuracy')
        # plt.title('Training and validation accuracy')
        # plt.legend(loc=0)
        # plt.figure()
        # plt.show()
        # img = io.BytesIO()
        # fig.savefig(img)
        # img.seek(0)
        # return send_file(img, mimetype = 'img/png')

        
      
        
@app.route('/grafik')
def grafik():
    import random
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    fig =Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    # return fig
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype = 'img/png')
   
       

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        preds = model_predict(img, model)
        target_names = ['Batu', 'Kista', 'Normal', 'Tumor']
        hasil_label = target_names[np.argmax(preds)]
        hasil_prob = "{:.2f}".format(100 * np.max(preds))
        return jsonify(result=hasil_label, probability = hasil_prob)

    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/training', methods=['GET', 'POST'])
def training():
    return render_template('training.html')

@app.route('/testing', methods=['GET', 'POST'])
def testing():
    return render_template('testing.html')

@app.route('/tentang', methods=['GET', 'POST'])
def tentang():
    return render_template('tentang.html')

if __name__ == '__main__':
    app.run(debug=True)