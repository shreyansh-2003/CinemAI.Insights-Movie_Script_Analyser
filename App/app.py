"""
Importing Required Libraries
"""

# Flask and Web-related libraries
from flask import Flask, render_template, redirect, url_for, jsonify, request
import glob 
import pandas as pd
from sklearn.utils import shuffle
from math import isnan
from flask_cors import CORS

from flask_caching import Cache

# Transformers library for natural language processing
import transformers
from transformers import RobertaTokenizer, TFRobertaModel
from transformers import BertTokenizer, TFBertForSequenceClassification

# TensorFlow and Keras for deep learning models
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

# Data/File Handling and Analytics Specific Libraries
import glob
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# Custom library/module for Movei On Click Analysis
import onclickMovie


"""
Initialising App
"""
app = Flask(__name__)
cache = Cache(app,config={'CACHE_TYPE': 'simple', 
                          'CACHE_DEFAULT_TIMEOUT': 3600})
CORS(app)



"""
Metadata prep
"""
#Checking whether the image in the dataset(metadata/scripts) exists within the poster set of images
image_locations = glob.glob("static/images/movie poster images/*.jpg")
image_imdb_ids = [int(image_locations_i.split('/')[-1].split('.jpg')[0]) for image_locations_i in image_locations]
movie_metadata = pd.read_csv('./Data/Raw/movie_metadata/movie_meta_data.csv')

"""
Image Data for Movie Analysis Page
"""
@app.route('/image_data')
@cache.cached(timeout=3600)
def load_image_data():
    #Create image data for each movie with valid IMDb ID
    image_data_glob = [{'imdbid': value[0], 
                    'src':"images/movie poster images/"+str(value[0])+".jpg", 
                    "title" : value[1],
                    "year" : value[3]} for value in movie_metadata.values if value[0] in image_imdb_ids]
    return image_data_glob


image_data_glob = load_image_data()

"""
BERT Tokenizer
"""
#Loading Tokenizer
tokenizer = BertTokenizer.from_pretrained('./Models/Pre-Trained/bert_tkz')



"""
Fitted Multilabel Binarizer
"""
# Load the saved MultiLabelBinarizer from the file
with open('./Models/mlb_model.pkl', 'rb') as file:
    mlb = pickle.load(file)

"""
Model Architecture: Movie Rating Prediction
"""
@app.route('/rating_model_cc')
@cache.cached(timeout=3600)
def load_rating_model(tokenizer):
    bert_model_rating = transformers.TFAutoModel.from_pretrained('./Models/Pre-Trained/bert_model', num_labels=1)
    input_ids_rating = tf.keras.layers.Input(shape = (512,), dtype = tf.int32, name = "input_ids") #Building the BERT model
    outputs_rating = bert_model_rating (input_ids_rating)[0]
    cls_token_rating = outputs_rating[:, 0, :]
    x_rating = tf.keras.layers.Dropout(0.50)(cls_token_rating) #adding dropout layer
    out = tf.keras.layers.Dense(10, activation = 'sigmoid')(x_rating) #Using a dense layer of 9 neurons as the number of unique categories is 9
    model_rating = tf.keras.Model(inputs = input_ids_rating, outputs = out)
    model_rating.compile(tf.keras.optimizers.Adam(learning_rate = 0.000001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model_rating.load_weights('./Models/bert_imdb_ratings_model.h5')

    return model_rating


"""
Model Architecture: Age Restriction Prediciton
"""
@app.route('/age_model_cc')
@cache.cached(timeout=3600)
def load_age_model(tokenizer):
    bert_model_age = TFBertForSequenceClassification.from_pretrained('./Models/Pre-Trained/bert_model_sequence', num_labels=1)
    input_ids_age = Input(shape=(512,), dtype=tf.int32, name="input_ids") #Building the BERT model
    outputs_age = bert_model_age(input_ids_age)
    model_age = Model(inputs=[input_ids_age], outputs=outputs_age.logits) #Building and compiling the overall model
    optimizer = Adam(learning_rate=0.000005)
    model_age.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    model_age.load_weights('./Models/bert_age_restrictions_model.h5')

    return model_age


"""
Model Architecture: Genre Prediction
"""
@app.route('/genre_model_cc')
@cache.cached(timeout=3600)
def load_genre_model(tokenizer):
    bert_model_genre = transformers.TFAutoModel.from_pretrained('./Models/Pre-Trained/bert_model', num_labels=1) #Building the BERT model
    input_ids_genre = tf.keras.layers.Input(shape = (512,), dtype = tf.int32, name = "input_ids")
    outputs_genre = bert_model_genre(input_ids_genre)[0]
    cls_token_genre = outputs_genre[:, 0, :]
    x_genre = tf.keras.layers.Dropout(0.50)(cls_token_genre) #adding dropout layer
    out_genre = tf.keras.layers.Dense(7, activation = 'sigmoid')(x_genre) #Using a dense layer of 7 neurons as the number of unique categories is 7. 
    model_genre = tf.keras.Model(inputs = input_ids_genre, outputs = out_genre)
    model_genre.compile(tf.keras.optimizers.Adam(learning_rate = 0.00001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model_genre.load_weights('./Models/bert_genre_model.h5')
    return model_genre

model_genre = load_genre_model(tokenizer)
model_age = load_age_model(tokenizer)
model_rating = load_rating_model(tokenizer)


"""
Home Page
"""
@app.route('/')
def index():
    shuffled_image_data = shuffle(image_data_glob)
    return render_template('index.html', image_data=shuffled_image_data)

"""
Movie Analysis Page
"""
# Movie Analysis: Route for handling the AJAX request and retrieving data
@app.route('/get_data/<int:imdb_id>')
def get_data(imdb_id):
    obj = onclickMovie.MovieAnalysis(str(imdb_id), show=False)

    if obj.corpus:
        image_data_ = [{'imdbid': value[0], 
                        'src':"images/movie poster images/"+str(value[0])+".jpg", 
                        "title" : value[1]} for value in movie_metadata.values if value[0]==imdb_id]
        
        movie_data = movie_metadata[movie_metadata['imdbid'] == imdb_id].to_dict('records')[0]
        movie_data['src'] = "images/movie poster images/"+str(imdb_id)+".jpg"
        parent_path = glob.glob("./Data/Raw/screenplay_data/data/raw_texts/raw_texts/*.txt")

        file_path = [path for path in parent_path if int(path.split('/')[-1].split('_')[1].split('.txt')[0]) == imdb_id][0]
        
        with open(file_path, 'r', encoding='utf-8') as file:
            script_content = file.read()

        movie_data = {key: value for key, value in movie_data.items() if not isinstance(value, float) or not isnan(value)}

        return render_template('movie_analysis.html', data=movie_data, image_data=image_data_, script_content=script_content)
    
    else:
        shuffled_image_data = shuffle(image_data_glob)
        return render_template('index.html', image_data=shuffled_image_data)

"""
Movie Rating Prediction
"""
# Movie Rating Prediction
@app.route('/movie_rating_prediction')
def movie_rating_prediction():
    return render_template('movie_rating_predict.html', image_data=image_data_glob, movie_metadata=movie_metadata)


# Getting Data From Movie Rating Prediction webpage
@app.route('/predict_movie_rating', methods=['POST'])
def predict_movie_rating():
    try:
        data = request.get_json()
        movie_script = data.get('movieScript', '')

        #Predicting Raating
        input_encoding = tokenizer(movie_script, truncation=True, padding='max_length', max_length=512, return_tensors='tf') #Tokenizing and encode the input text
        prediction = model_rating.predict(x={'input_ids': input_encoding['input_ids']})
        predicted_rating = np.argmax(prediction)
        return str(predicted_rating)+"/10" 

    except Exception as e:
        return str(e)

"""
Movie Rating Prediction
"""
   
# Age Restriction Prediction
@app.route('/age_restriction_prediction')
def age_restriction_prediction():
    return render_template('movie_age_restrict_prediction.html', image_data=image_data_glob, movie_metadata=movie_metadata)


@app.route('/predict_age_restriction', methods=['POST'])
#User Defined Function to predict age restriction based on text input
def predict_age_restriction():
    try:
        data = request.get_json()
        movie_script = data.get('movieScript', '')

        input_encoding = tokenizer(movie_script, truncation=True, padding='max_length', max_length=512, return_tensors='tf') #Tokenizing and encode the input text
        prediction = model_age.predict(x={'input_ids': input_encoding['input_ids']}) #Making prediction
        predicted_age = np.squeeze(prediction) #Extracting the predicted rating (assuming the model is designed for regression)

        return str(predicted_age) 

    except Exception as e:
        return str(e)



"""
Movie Rating Prediction
"""
# Movie Genre Prediciton
@app.route('/genre_prediction')
def genre_prediction():
    return render_template('movie_genre_prediction.html', image_data=image_data_glob, movie_metadata=movie_metadata)


@app.route('/predict_movie_genre', methods=['POST'])
#User Defined Function to predict genre based on text input
def predict_movie_genre():
    try:
        data = request.get_json()
        movie_script = data.get('movieScript', '')
        tokenized_text = tokenizer(movie_script, padding='max_length', truncation=True, return_tensors='tf',max_length=512)
        predictions = model_genre.predict(tokenized_text['input_ids'])
        threshold = 0.5
        binary_predictions = (predictions>threshold).astype(int)
        predicted_labels = mlb.inverse_transform(binary_predictions)[0]
        final = ', '.join(predicted_labels)
        print(final)
        return str(final)

    except Exception as e:
        return str(e)
    

"""
General Functions for Dropdown Autofill
"""
#Function to get script content based on selected movie title
def get_script_content(selected_title):
    imdb_id = movie_metadata[movie_metadata['title'] == selected_title]['imdbid'].values[0]

    parent_path = glob.glob("./Data/Raw/screenplay_data/data/raw_texts/raw_texts/*.txt")
    file_path = [path for path in parent_path if int(path.split('/')[-1].split('_')[1].split('.txt')[0]) == imdb_id][0]

    with open(file_path, 'r', encoding='utf-8') as file:
        script_content = file.read()

    return script_content

#New route to handle the AJAX request for fetching script content
@app.route('/get_script_content', methods=['POST'])
def get_script_content_route():
    try:
        data = request.get_json()
        selected_title = data.get('selected_title', '')

        script_content = get_script_content(selected_title)

        return jsonify({'script_content': script_content})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)

    

