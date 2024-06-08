""" 
Importing Required Libraries
""" 
#File handling and system operations
import os
import glob

#Numeric and data manipulation
import numpy as np
import pandas as pd

#Text and language processing
import re
from nltk import tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import pos_tag

#Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot
from plotly import tools
from plotly.subplots import make_subplots
import chart_studio.plotly as py

#Natural Language Processing (NLP)  (For Topic Modelling)
from gensim import corpora, models

#Word cloud generation
from wordcloud import WordCloud

#Computer vision
import cv2

#Spacy for advanced NLP tasks
import spacy

#Additional libraries for data analysis and visualization
from collections import Counter


class MovieAnalysis:

    def __init__(self,imdb_id,show):
        self.imdb_id = imdb_id
        self.character_scripts_locations = glob.glob('./Data/Raw/character_scripts/*/')
        self.movie_name, self.df= self.create_movie_dataset()

        try:
            self.corpus = " ".join(self.df['content'])
            self.run_all(show=show)
        except:
            self.corpus = False
        

    # Dataset Creation
    def create_movie_dataset(self):
        """ 
        The function takes three parameters:
        1. imdb_id: a string id
        2. character_scripts_locations: he location of the movie scripts files
        3. script an optional parameter with a default value of False. If True the returned value of the funciton will be the entrie movie's script as a corpus.

        => Proces

        1. The function initializes an empty list called ```character_scripts``` and an empty list called ```data```.

        2. It iterates over the provided character_scripts_locations, extracts the IMDb ID from the file paths, and checks if it matches the specified imdb_id. If there is a match, it retrieves the movie name and locates character scripts using the glob module.

        3. For each character script file, it reads the content line by line. Each line is then processed to extract information such as segment, segment_scene, label, and text. This information is stored in a dictionary called character_dict, which is appended to the data list.

        4. The code attempts to create a pandas DataFrame (df) (try-except-catch) from the collected data. If the optional parameter script is True, it processes the DataFrame to sort and concatenate the script lines into a single string (entire_script). The function returns either the DataFrame or the entire concatenated script, depending on the value of the script parameter.

        If an exception occurs during the DataFrame creation, the function returns None.
        """
        character_scripts = []

        data = []

        for movies in self.character_scripts_locations:
            id = movies.split('/')[-2].split('_')[1] #Eg. File Name...

            if id[0]=='0':
                id = id[1:]
            
            if id[0]=='0' and id[1]=='0':
                id = id[2:]

            if id[0]=='0' and id[1]=='0' and id[2]=='0':
                id = id[3:]

                
            if id == self.imdb_id:
                movie_name = movies.split('/')[-2].split('_')[0]
                character_scripts = glob.glob(movies + '*.txt')

        for character in character_scripts:
            character_name = character.split('/')[-1].split('_')[0]

            with open(character,'r') as file:
                lines = file.readlines()

            for line_i in lines:
                segment = line_i.split(')')[0]
                segment_scene = line_i.split(')')[1]
                label = line_i.split(')')[2].split(':')[0].strip()

                text_ = line_i.split(')')[2].split(':')[1:]
                text = ''
                for words in text_:
                    text += words
                text = text.strip()

                character_dict = {'character' : character_name,
                            'segment' : segment,
                            'segment_scene': segment_scene,
                            'kind': label,
                            'content': text}
            
                data.append(character_dict)


        
        try: 
            df = pd.DataFrame.from_dict(data)
        
        except:
            return False, False
        
        try:
            return movie_name, df
        
        except:
            return False, False
        
    # Scene-Wise Sentiment Analyser
    def film_sentiment(self, colour, show = True):
        """
        Analyzing and visualizing sentiment across scenes in the movie.

        Parameters:
        - df_movie: DataFrame containing movie data.
        - moviename: Name of the movie.
        - colour: Color for the sentiment plot.

        Returns:
        - df_sentiment: DataFrame containing sentiment scores for each scene.
        """
        analyzer = SentimentIntensityAnalyzer()
        sc_sent = {}

        for x in range(len(self.df)):
            scene = re.sub(r"[^a-zA-Z0-9.? ]+", '', self.df.content[x])
            scene_sentence = tokenize.sent_tokenize(scene)

            # Check if the length of scene_sentence is not zero to avoid ZeroDivisionError
            if len(scene_sentence) > 0:
                sentiments = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
                for sentence in scene_sentence:
                    vs = analyzer.polarity_scores(sentence)
                    sentiments['compound'] += vs['compound']
                    sentiments['neg'] += vs['neg']
                    sentiments['neu'] += vs['neu']
                    sentiments['pos'] += vs['pos']

                sentiments['compound'] = sentiments['compound'] / float(len(scene_sentence))
                sentiments['neg'] = sentiments['neg'] / float(len(scene_sentence))
                sentiments['neu'] = sentiments['neu'] / float(len(scene_sentence))
                sentiments['pos'] = sentiments['pos'] / float(len(scene_sentence))
                dic = 'scene_' + str(x)
                sc_sent[dic] = sentiments
            else:
                # Handle the case where scene_sentence is empty
                dic = 'scene_' + str(x)
                sc_sent[dic] = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}

        sents = [sc_sent[keys] for keys in sc_sent]
        df_sentiment = pd.DataFrame(sents)
        df_zero = pd.DataFrame(0, df_sentiment.index, columns=['Zero'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sentiment.index, y=df_sentiment['compound'], mode='lines',
                                name='Average Sentiment', line=dict(color=colour)))
        fig.add_trace(go.Scatter(x=df_zero.index, y=df_zero['Zero'], mode='lines', name='Zero line',
                                line=dict(color='crimson', dash='dot')))
        fig.update_layout(title=dict(text = '<b> Sentiment across the ' + self.movie_name + ' Movie <b>', x =0.5),
                        xaxis_title='<b> Scenes <b>', yaxis_title='<b> Average Sentiments <b>',
                          showlegend=True)
        
        self.apply_custom_styles(fig)
        fig.write_html('templates/graphs/scenewise_sentiment.html')

        if show:
            fig.show()
            
        else:
            return
        
    # Word Cloud & Topic Modelling
    def perform_topic_modeling(self, num_topics=3, num_words_per_topic=50, show=True):
        """
        Performing topic modeling on the 'content' column of the scripts DataFrame.

        Parameters:
        - num_topics: Number of topics for LDA modeling.
        - num_words_per_topic: Number of words to display in each word cloud.
        - show: Whether to display the word clouds.

        Returns:
        - None (displays word clouds for each topic).
        """

        # Concatenate the content column to create a corpus
        corpus = " ".join(self.df['content'])

        # Tokenize the corpus
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(corpus)
        filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

        # Create a dictionary and a document-term matrix
        dictionary = corpora.Dictionary([filtered_words])
        corpus_ = [dictionary.doc2bow(filtered_words)]


        # Perform topic modeling using LDA
        lda_model = models.LdaModel(corpus_, num_topics=num_topics, id2word=dictionary, passes=15)

        custom_mask = np.array(cv2.imread('./Data/cinema_mask.png'))

        # Generate word clouds for each topic with the custom mask
        for topic_id in range(num_topics):
            topic_words = [word for word, _ in lda_model.show_topic(topic_id, topn=num_words_per_topic)]
            topic_text = " ".join(topic_words)

            wordcloud = WordCloud(width=800, height=800,
                                mode='RGBA',  # Set mode to RGBA
                                background_color=(0, 0, 0, 0),  # Specify transparent black background
                                mask=custom_mask).generate(topic_text)

            # Convert the word cloud image to Plotly format
            img_array = np.array(wordcloud.to_image())
            fig = px.imshow(img_array)
            fig.update_layout(title_text=f'<b>Topic {topic_id + 1}</b>', title=dict(x=0.5), showlegend=False)

            # Remove axis
            fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
            self.apply_custom_styles(fig)
            fig.write_html(f'templates/graphs/topic_cloud_{topic_id + 1}.html')

            if show:
                fig.show()
            else:
                pass



    # NER Charts
    def perform_ner_and_plot(self, show=True):
        # Performing Named Entity Recognition (NER) using spaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.corpus)

        #Extracting the entities and their labels
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        #Createing a DataFrame from the entities
        entities_df = pd.DataFrame(entities, columns=["Entity", "Label"])

        #Bar chart of entity labels
        label_counts = entities_df["Label"].value_counts()
        fig1 = px.bar(label_counts, x=label_counts.index, y=label_counts.values,
                    labels={'x': 'Entity Label', 'y': 'Count'},
                    color=label_counts.index)
        
        fig1.update_layout(title=dict(text = '<b> Named Entity Recognition: Entity Label Distribution <b>', x =0.5))

        #Word cloud of entities
        wordcloud_data = entities_df.groupby("Entity").size().reset_index(name='Count')
        fig2 = px.scatter(wordcloud_data, x='Entity', y='Count',
                        labels={'Entity': 'Entity', 'Count': 'Count'},
                        size='Count', color='Count')
        fig2.update_layout( title = dict(text='<b>Named Entity Recognition: Entity Word Cloud</b>', x=0.5))

        self.apply_custom_styles(fig1)
        self.apply_custom_styles(fig2)
        fig1.write_html('templates/graphs/ner1.html')
        fig2.write_html('templates/graphs/ner2.html')

        if show:
            #Showing both the plots !
            fig1.show()
            fig2.show()

            title=dict(text = '<b> Named Entity Recognition: Entity Word Cloud <b>', x =0.5),

        else: 
            return

    # POS Tags Chart
    def pos_tagging_and_create_chart(self,show=True):
        #Mapping of POS Generated short forms to full-forms
        pos_tag_mappings = {
                        'CC': 'Coordinating conjunction',
                        'CD': 'Cardinal number',
                        'DT': 'Determiner',
                        'EX': 'Existential there',
                        'FW': 'Foreign word',
                        'IN': 'Preposition or subordinating conjunction',
                        'JJ': 'Adjective',
                        'JJR': 'Adjective, comparative',
                        'JJS': 'Adjective, superlative',
                        'LS': 'List item marker',
                        'MD': 'Modal',
                        'NN': 'Noun, singular or mass',
                        'NNS': 'Noun, plural',
                        'NNP': 'Proper noun, singular',
                        'NNPS': 'Proper noun, plural',
                        'PDT': 'Pre determiner',
                        'POS': 'Possessive ending',
                        'PRP': 'Personal pronoun',
                        'PRP$': 'Possessive pronoun',
                        'RB': 'Adverb',
                        'RBR': 'Adverb, comparative',
                        'RBS': 'Adverb, superlative',
                        'RP': 'Particle',
                        'S': 'Simple declarative clause',
                        'SBAR': 'Clause introduced by a subordinating conjunction',
                        'SBARQ': 'Direct question introduced by a wh-word or wh-phrase',
                        'SINV': 'Inverted declarative sentence',
                        'SQ': 'Inverted yes/no question',
                        'SYM': 'Symbol',
                        'VBD': 'Verb, past tense',
                        'VBG': 'Verb, gerund or present participle',
                        'VBN': 'Verb, past participle',
                        'VBP': 'Verb, non-3rd person singular present',
                        'VBZ': 'Verb, 3rd person singular present',
                        'WDT': 'Wh-determiner',
                        'WP': 'Wh-pronoun',
                        'WP$': 'Possessive wh-pronoun',
                        'WRB': 'Wh-adverb'
                    }

        #Tokenizing the corpus
        words = word_tokenize(self.corpus)
        tagged = pos_tag(words)

        #Creating a customized bar chart of POS tags
        tag_freq = FreqDist(tag for (word, tag) in tagged)
        top_tags = tag_freq.most_common(10)
        top_tags = [(pos_tag_mappings.get(tag, tag), count) for tag, count in top_tags]

        #Creating a bar chart using Plotly
        fig = px.bar(
            top_tags,
            x=1,  # Counts
            y=0,  # POS Tags
            orientation='h',  # Horizontal bar chart
            color=0,  # Color based on POS Tags
            labels={'0': '<b>POS Tags<b>', '1': '<b>Frequency<b>'}
        )

        fig.update_layout(title = dict(text='<b>Top 10 POS Tags<b>', x =0.5))

        self.apply_custom_styles(fig)
        fig.write_html("templates/graphs/pos.html")

        if show:
            #Showing the plot
            fig.show()
        else:
            return

    def chracter_count_analysis(self, show=True):
        """
        Performing character interaction analysis for the given movie.
        
        Parameters:
        - df: DataFrame containing movie scenes and characters.
        - movie_name: Name of the movie.
        
        Returns:
        - List of characters used in the analysis.
        """

        def remove_unwanted_characters(df):
            """
            Removing unwanted characters from the DataFrame.
            
            Parameters:
            + df: DataFrame containing the character data.
            
            Returns:
            + List of characters that appeared in both 'text' and 'dialog' kinds.
            """
            direct_characters = df[df['kind'] == 'text']['character'].unique()
            dialog_characters = df[df['kind'] == 'dialog']['character'].unique()
            actual_characters = [character for character in direct_characters if character in dialog_characters]
            return actual_characters

        def count_character_appearances(df):
            """
            Counting the number of times each character appears in the DataFrame.
            
            Parameters:
            - df: DataFrame containing the character data.
            
            Returns:
            - Dictionary with character counts.
            """
            character_count = dict(Counter(df['character']))
            
            # Remove characters with only one appearance
            character_count = {k: v for k, v in character_count.items() if v > 1}
            
            return character_count

        def plot_character_appearances(character_count, movie_name):
            """
            Plotingt the appearances of characters in the movie script.
            
            Parameters:
            - character_count: Dictionary with character counts.
            - movie_name: Name of the movie.
            """
            df_character_count = pd.DataFrame(character_count.items(), columns=['Characters', 'counts']).sort_values(by='counts')
            
            fig = px.bar(df_character_count, x='counts', y='Characters', orientation='h',
                        hover_data=df_character_count.columns, color='counts',
                        labels={'counts': '<b> Character Counts <b>'}, width=1000, height=1000)
            
            fig.update_layout(title=dict(text=f'<b> Number of Times Characters appeared in the {movie_name} Movie <b>',x=0.5),
                            xaxis_title='<b> Counts <b>', yaxis_title='<b> Characters <b>')
            
            self.apply_custom_styles(fig)
            fig.write_html('templates/graphs/scenes_per_character.html') 

            if show:
                iplot(fig)

            else:
                return

        def count_scenes_per_character(df, characters):
            """
            Counting the number of scenes each character appeared in.
            
            Parameters:
            - df: DataFrame containing the character data.
            - characters: List of characters.
            
            Returns:
            - Dictionary with character counts per scene.
            """
            characters_per_scene = {}
            for character in characters:
                count = df[df['character'] == character]['segment_scene'].nunique()
                characters_per_scene[character] = count
                
            return characters_per_scene

        def plot_scenes_per_character(characters_per_scene, movie_name):
            """
            Plotting the number of scenes each character appeared in.
            
            Parameters:
            - characters_per_scene: Dictionary with character counts per scene.
            - movie_name: Name of the movie.
            """
            df_per_scene = pd.DataFrame(characters_per_scene.items(), columns=['Characters', 'Scene counts']).sort_values(by='Scene counts')
            
            fig = px.bar(df_per_scene, x='Scene counts', y='Characters', orientation='h',
                        hover_data=df_per_scene.columns, color='Scene counts',
                        labels={'Scene counts': '<b> Scene Counts <b>'}, width=1000, height=900,
                        color_continuous_scale=px.colors.sequential.speed)
            
            fig.update_layout(title=dict(text=f'<b> Number of Scenes Each Character Spoke In, in the {movie_name} movie <b>',x=0.5),
                            xaxis_title='<b> Scene counts <b>', yaxis_title='<b> Characters <b>')
            

            self.apply_custom_styles(fig)
            
            fig.write_html('templates/graphs/dlg_per_character.html') 
            
            if show:
                iplot(fig)

            else:
                return

        #Extracting the  characters common in both 'text' and 'dialog' kinds
        actual_characters = remove_unwanted_characters(self.df)
        
        #Counting the number of times each character appears in the movie script
        character_count = count_character_appearances(self.df)
        
        #Plotting character appearances in the movie script
        plot_character_appearances(character_count, self.movie_name)
        
        #Counting the number of scenes each character appeared in
        characters_per_scene = count_scenes_per_character(self.df, actual_characters)
        
        #Plotting the number of scenes each character appeared in
        plot_scenes_per_character(characters_per_scene, self.movie_name)
        
        return
    
    # Scene-wise NRC Sentiment-Emotions Plot
    def film_emotional_arc(self,show=True):
        """
        Analyzes and visualizes emotional arcs across scenes in a movie.

        Parameters:
        - df_movie: DataFrame containing movie data.

        Returns:
        - df_scene_emotions: DataFrame containing emotional scores for each scene.
        """
        def cap_sentence(s):
            return re.sub("(^|\s)(\S)", lambda m: m.group(1) + m.group(2).upper(), s)

        df_contents = self.df[['segment', 'content']]
        df_emotions = pd.read_csv('./Models/NRC Word-Emotion Association Lexicon/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                                names=["word", "emotion", "association"], sep='\t')
        df_emotion_word = df_emotions.pivot(index='word', columns='emotion', values='association').reset_index()
        emotions = df_emotion_word.columns.drop('word').tolist()
        df_emo = pd.DataFrame(0, df_contents.index, columns=emotions)
        stemmer = SnowballStemmer("english")

        for x in range(len(df_contents)):
            scene_conts = re.sub(r"[^a-zA-Z0-9 ]+", '', df_contents.content[x])
            doc = word_tokenize(scene_conts)
            for word in doc:
                word = stemmer.stem(word.lower())
                emotion_score = df_emotion_word[df_emotion_word.word == word]
                if not emotion_score.empty:
                    for emotion in emotions:
                        df_emo.at[x, emotion] += emotion_score[emotion]

        df_scene_emotions = pd.concat([df_contents, df_emo], axis=1)
        df_scene_emotions['word_count'] = df_scene_emotions['content'].apply(tokenize.word_tokenize).apply(len)
        for emotion in emotions:
            df_scene_emotions[emotion] = df_scene_emotions[emotion] / df_scene_emotions['word_count']

        fig = make_subplots(rows=5, cols=2, subplot_titles=(
            cap_sentence(emotions[0]), cap_sentence(emotions[1]), cap_sentence(emotions[2]),
            cap_sentence(emotions[3]), cap_sentence(emotions[4]), cap_sentence(emotions[5]),
            cap_sentence(emotions[6]), cap_sentence(emotions[7]),
            cap_sentence(emotions[8]), cap_sentence(emotions[9]))
                        )

        row = 0
        count = 0
        for x in emotions:
            if count % 2:
                fig.add_trace(go.Scatter(x=df_scene_emotions.index, y=df_scene_emotions[x]), row=row, col=2)
            else:
                row += 1
                fig.add_trace(go.Scatter(x=df_scene_emotions.index, y=df_scene_emotions[x]), row=row, col=1)
            count += 1

            self.apply_custom_styles(fig)

            fig.update_xaxes(title_text='Scenes', dtick=20,linecolor='rgba(173, 216, 230, 0.2)', gridcolor='rgba(173, 216, 230, 0.2)')
            fig.update_yaxes(title_text="Average Sentiment",linecolor='rgba(173, 216, 230, 0.2)', gridcolor='rgba(173, 216, 230, 0.2)')
            fig.update_layout(height=1500, width=1000,
                            title_text="<b> Emotional arcs identified across the scenes <b>",
                            showlegend=False,
                            title=dict(x=0.5))
            
        fig.write_html('templates/graphs/ncr_emotion_plot.html')
        
        
        if show:
            fig.show()

        else: 
            return

    # Characters Specific Scene-wise NRC Sentiment-Emotions Plot
    def emotional_arc_character_plot(self, character, show = True):
        """
        Analyzes and visualizes emotional arcs of a character across scenes in a movie.

        Parameters:
        - df_movie: DataFrame containing movie data.
        - character: Name of the character.

        Returns:
        - df_xter_emotions: DataFrame containing emotional scores for the character in each scene.
        """
        def cap_sentence(s):
            return re.sub("(^|\s)(\S)", lambda m: m.group(1) + m.group(2).upper(), s)

        def xter_count_perscene(df, characters):
            sc_xters = []
            sc_dia = []

            for x in range(len(df)):
                sc_xtrs = []
                sc_di = []

                if df['character'][x] == character:
                    segment_scene = df['segment_scene'][x]
                    content = df['content'][x]

                    sc_xtrs.append(character)
                    sc_di.append(content)

                    sc_xters.append(sc_xtrs)
                    sc_dia.append(sc_di)
                else:
                    sc_xters.append(0)  # Set 0 for scenes where the character is not present
                    sc_dia.append(0)    # Set 0 for scenes where the character is not present

            # Count the appearance of the character in each scene
            sc_cts = [1 if x == character else 0 for x in df['character']]
            df_counts = pd.DataFrame(sc_cts, columns=[character])

            df_scene_dialogue = pd.DataFrame(list(zip(sc_xters, sc_dia)), columns=['characters', 'dialogues'])
            return df_counts, df_scene_dialogue


        df_cts, df_xt = xter_count_perscene(self.df, character)

        df_emotions = pd.read_csv('./Models/NRC Word-Emotion Association Lexicon/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                                names=["word", "emotion", "association"], sep='\t')

        df_emotion_word = df_emotions.pivot(index='word', columns='emotion', values='association').reset_index()
        emotions = df_emotion_word.columns.drop('word').tolist()
        df_emo = pd.DataFrame(0, df_xt.index, columns=emotions)
        stemmer = SnowballStemmer("english")
        df_xt['dialogues'] = df_xt.apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', str(x['dialogues'])).lower(), axis=1)

        for x in range(len(df_xt)):
            scene_contents = df_xt['dialogues'][x]
            doc = word_tokenize(scene_contents)
            for word in doc:
                word = stemmer.stem(word.lower())
                emotion_score = df_emotion_word[df_emotion_word.word == word]
                if not emotion_score.empty:
                    for emotion in emotions:
                        df_emo.at[x, emotion] += emotion_score[emotion]

        df_xter_emotions = pd.concat([df_xt, df_emo], axis=1)
        df_xter_emotions['word_count'] = df_xter_emotions['dialogues'].apply(tokenize.word_tokenize).apply(len)

        df_xter_emotions = df_xter_emotions.fillna(0)

        for emotion in emotions:
            df_xter_emotions[emotion] = df_xter_emotions[emotion] / df_cts[character]

            df_xter_emotions = df_xter_emotions.fillna(0)


            fig = make_subplots(rows=5, cols=2, subplot_titles=(
                cap_sentence(emotions[0]), cap_sentence(emotions[1]), cap_sentence(emotions[2]),
                cap_sentence(emotions[3]), cap_sentence(emotions[4]), cap_sentence(emotions[5]),
                cap_sentence(emotions[6]), cap_sentence(emotions[7]),
                cap_sentence(emotions[8]), cap_sentence(emotions[9]))
            )


            row = 0
            count = 0
            for x in emotions:
                if count % 2:
                    fig.add_trace(go.Scatter(x=df_xter_emotions.index, y=df_xter_emotions[x]), row=row, col=2)
                else:
                    row += 1
                    fig.add_trace(go.Scatter(x=df_xter_emotions.index, y=df_xter_emotions[x]), row=row, col=1)
                count += 1
                fig.update_xaxes(title_text='Scenes', dtick=20,linecolor='rgba(173, 216, 230, 0.2)', gridcolor='rgba(173, 216, 230, 0.2)')
                fig.update_yaxes(title_text="Average Sentiment",linecolor='rgba(173, 216, 230, 0.2)', gridcolor='rgba(173, 216, 230, 0.2)')

            fig.update_xaxes(title_text='Scenes', dtick=20,linecolor='rgba(173, 216, 230, 0.2)', gridcolor='rgba(173, 216, 230, 0.2)')
            fig.update_yaxes(title_text="Average Sentiment",linecolor='rgba(173, 216, 230, 0.2)', gridcolor='rgba(173, 216, 230, 0.2)')
            fig.update_layout(height=1500, width=1000,
                            title_text="<b> Emotional arcs identified across the scenes <b>",
                            showlegend=False,
                            title=dict(x=0.5))
                

        # fig.write_html('templates/graphs/ncr_emotion_plot.html')

        if show:
            fig.show()
        else:
            return
        

    def run_all(self,show):

        self.film_sentiment('purple', show)
        self.perform_topic_modeling(num_topics=3, num_words_per_topic=50,show=show)
        self.perform_ner_and_plot(show)
        self.pos_tagging_and_create_chart(show)
        self.chracter_count_analysis(show)
        self.film_emotional_arc(show)
        # self.emotional_arc_character_plot(character, show =show)

    def apply_custom_styles(self, fig):
        """
        Applying custom styles to the given Plotly figure.

        Parameters:
        - fig: Plotly figure to apply custom styles.
        """
        # Add your custom styles here
        fig.update_layout(
            paper_bgcolor='black',  # Background color of the plot area
            plot_bgcolor='black',   # Background color of the entire figure
            font=dict(color='white'),
        )

        # Apply styles to each axis
        for axis_name in ['xaxis', 'yaxis']:
            fig.update_layout({f'{axis_name}': {
                'tickfont': dict(color='white'),  # Light blue with low opacity
                'linecolor': 'rgba(173, 216, 230, 0.2)',  # Light blue border color with low opacity
                'gridcolor': 'rgba(173, 216, 230, 0.2)',  # Light blue grid color with lower opacity
                'title': {'font': {'color': 'white'}},  # X-axis and Y-axis title color
                'tickmode': 'array',
                'titlefont': dict(color='white'),  # X-axis and Y-axis title color
                'tickcolor': 'rgba(173, 216, 230, 0.2)'  # Light blue tick color with low opacity
            }})
