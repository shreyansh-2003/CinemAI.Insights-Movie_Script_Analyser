> # CinemAI Insights - A One Stop Solution

CinemAI is a one stop solution solution created for movie buffs. The fullstack app has a movie scene-wise and character-wise analysis feature. It also has genre classification, movie, movie rating prediction and age restriction prediction based on a movie's script.

**Running the CinemAI Insights Application**

- Downloading additional (Large) Files
  1. Download the *.H5 files*, *pre-trained tokenizers* and *NCR-lexion* into the ```Models Folder``` from the [**reostiory link**](https://1drv.ms/f/s!AqM-iZWYLD9iiMEmcCXRkrZWoTSUtQ?e=J1UXvK) givem in README.md of the Models folder.
  2. Download the *Raw Folder*, *Processed Folder* and *cinema_mask.png* into the ```Data Folder``` from the [**reostiory link**](https://onedrive.live.com/?authkey=%21ADXkviDnwGJw8%2DE&id=623F2C9895893EA3%21139432&cid=623F2C9895893EA3) givem in README.md of the Data folder.
  3. Extract the compressed file ```movie poster images.zip``` present in *App/static/images*

- Install required libraries from requirements.txt (Command -> pip install -r requirements.txt)

- Run app.py in ```App FOlder``` as a Flask server

---

> ## Home Page

The home page offers a wide range of movie options to choose from. Each movie poster can be resized to small, medium, large, or extra-large.

![img](https://github.com/shreyansh-2003/CinemAI.Insights-Movie_Script_Analyser/assets/105413094/d42f5615-1bfc-4cdf-a688-0c87f83ec03e)

---

> ## Plot Pulse: Movie Analysis - OnClick()

When a movie poster is clicked, an in-depth analysis is generated within seconds. The analysis page is divided into two halves:

- **Left Side**:
  - Contains the movie script.
  - Includes a metadata button that, when clicked, replaces the movie script with details such as budget, rating, cast, awards, etc.

- **Right Side**:
  - Showcases various NLP techniques applied to the movie script, represented through interactive visualizations:
    1. **Unveiling Movie Themes with Topic Modeling**: Generates word clouds of prevalent topics in the shape of Melpomene and Thalia Theatre masks.
    2. **Feel the Vibes: Vader Scene-wise Sentiment**: Analyzes the emotional tone of the script on a scene-by-scene basis using sentiment analysis.
    3. **Spotlight on Characters and Locations with NER**: Identifies and classifies entities such as characters, locations, and organizations within the script using Named Entity Recognition (NER).
    4. **A Character's Stage: Scenes per Character**: Shows the frequency of each character's appearances throughout the movie.
    5. **Words That Matter: Dialogue per Character**: Highlights how often each character speaks in the movie.
    6. **Emotion Unveiled: Scene-wise NRC Lexicon Analysis**: Provides an emotional landscape of the script with NRC Lexicon Analysis, covering emotions like anger, anticipation, disgust, fear, joy, positive, negative, sadness, surprise, and trust.
    7. **Decoding Language: POS Tags**: Analyzes the parts of speech used in the script.



https://github.com/shreyansh-2003/CinemAI.Insights-Movie_Script_Analyser/assets/105413094/3535e050-e8c1-4b68-a780-313e443e9339

---

> ## BERTalizing Movie Scripts: Script-based Classification and Predictions

There are three distinct BERT (Bidirectional Encoder Representations from Transformers) based applications that have been fine-tuned to predict the genre (multi-label), age restriction (numeric value), and IMDb rating (numeric value) based on movie scripts. Users can either type in a script in an empty box or autofill the box with a selection of movies from the dropdown menu.

> #### Genre Classification based on Movie Script

<img width="1267" alt="Screenshot 2024-06-22 at 5 11 35 PM" src="https://github.com/shreyansh-2003/CinemAI.Insights-Movie_Script_Analyser/assets/105413094/f73c4944-3fb6-41c8-bf0c-1c31cecbd20a">

> #### IMDB Rating Prediction based on Movie Script 
<img width="1347" alt="Screenshot 2024-06-22 at 5 10 52 PM" src="https://github.com/shreyansh-2003/CinemAI.Insights-Movie_Script_Analyser/assets/105413094/27dc92b5-b704-4e18-8ac6-de985001366f">

> #### Movie Age Restriction based on Movie Script
<img width="1303" alt="Screenshot 2024-06-22 at 5 12 32 PM" src="https://github.com/shreyansh-2003/CinemAI.Insights-Movie_Script_Analyser/assets/105413094/d0a7ab51-bc99-4269-aa6d-7b8914827536">

---

