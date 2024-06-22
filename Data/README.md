> # CinemAi Insights - Data Files

The data files necessary to run the project are over 1 GB in size and are not uploaded directly to the repository. To run the application, **download** the required files from the link below.

[Data Repository Link](https://1drv.ms/f/s!AqM-iZWYLD9iiMEoNeS-IOfAYnDz4Q?e=PvThzu)

--

> ## Folders Required

### Processed
- `genre_dataset.csv`
- `ratings_df.csv`
- `age_dataset.csv`

### Raw
- `screenplay_data` (Folder)
- `character_scripts` (Folder)
- `movie_metadata` (Folder)

### Additional File
- `cinema_mask.png`

--

> ## Usage

- The `Processed` folder is used as a target class when training the dataset and is necessary for running the `.ipynb` notebooks in the notebooks folder. It was also used to map the movie poster images with the movie scripts while analyzing and predicting on movies.
  
- The `Raw` folder is used as input for performing analysis when movie posters are clicked on the home page or selected from the dropdown.

- The `cinema_mask.png` is used as a mask to generate the topic word clouds in the shape of theatrical masks while generating visualizations for the Plot Pulse - Movie Analysis Page.

--
