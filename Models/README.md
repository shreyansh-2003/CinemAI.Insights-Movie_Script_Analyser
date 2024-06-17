# CinemAI Insights - A One Stop Solution (Models)

As the fine-tuned and trained models are very large files and couldn't be uploaded using LFS Git, please find the models in the link attached below.

*[Model Repo Link](https://1drv.ms/f/s!AqM-iZWYLD9iiMEmcCXRkrZWoTSUtQ?e=J1UXvK)* <br>

The files in the folder link above include:

+ **NRC Word-Emotion Association Lexicon**: Developed by the National Research Council Canada (NRC) under the supervision of Dr. Saif Mohammad, to help in understanding and analyzing the emotional content of textual data. These weights and rules are used in the application for analysing different emotions in movie scenes.

<br>

+ **Pre-Trained Folder**  
  + BERT Tokenizer: The BERT tokenizer is responsible for converting raw text into a format that the BERT model can process. It tokenizes the text into subwords or tokens.
  + BERT Model: BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed to understand the context of a word in search queries
  + BERT Sequence Model: This refers to using the BERT model for tasks involving sequences of text, such as sentence classification or token classification.

<br>

+ **bert_age_restrictions_model.h5**: These are the fine-tuned BERT model weights used for predicting the **probable age restriction** (numerical age limit) of a movie based on its movie script.

<br>

+ **bert_genre_model.h5**: These are the fine-tuned BERT model weights used for classifying the **probable genre/s** of a movie based on its movie script.

<br>

+ **bert_imdb_ratings_model.h5**: These are the fine-tuned BERT model weights used for predicting the **probable IMDB Rating** (out of 10.0) of a movie based on its movie script.

