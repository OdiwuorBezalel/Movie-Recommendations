# Movie Recommender System

## Overview
This project implements a hybrid movie recommender system using the MovieLens dataset. It combines content-based filtering and collaborative filtering techniques to provide personalized movie recommendations.

## Features
- *Content-Based Filtering*: Recommends movies based on metadata such as titles and genres.
- *Collaborative Filtering*: Uses user-movie interaction data to identify similar users or movies.
- *Hybrid Approach*: Merges both techniques for better recommendation accuracy.
- *Dataset Used*: MovieLens dataset (small version).



## Usage
1. *Download and Extract the Dataset*
   python
   !curl http://files.grouplens.org/datasets/movielens/ml-latest-small.zip -o ml-latest-small.zip
   import zipfile
   with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
       zip_ref.extractall()
   

2. *Preprocess the Data*
   python
   import pandas as pd
   import re
   
   movies = pd.read_csv('ml-latest-small/movies.csv')
   def clean_title(title):
       return re.sub("[^a-zA-Z0-9 ]", "", title)
   movies['clean_title'] = movies['title'].apply(clean_title)
   

3. *Run the Recommender System*
   bash
   python main.py
   
   This will train the models and generate movie recommendations for users.

## Project Structure

movie-recommender/
│── data/              # Dataset storage
│── models/            # Trained models and utilities
│── src/
│   ├── content_based.py    # Content-based filtering implementation
│   ├── collaborative.py    # Collaborative filtering implementation
│   ├── hybrid.py           # Hybrid recommender
│   ├── utils.py            # Utility functions
│── main.py           # Main script to run the recommender
│── config.py         # Configuration settings
│── requirements.txt  # Dependencies
│── README.md         # Project documentation


## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Surprise (for collaborative filtering)
- Flask (optional, for API deployment)

## Evaluation
The system supports evaluation using:
- *RMSE (Root Mean Square Error)* for rating predictions.
- *Precision@K and Recall@K* for top-k recommendations.

To run the evaluation:
bash
python evaluate.py


## Future Improvements
- Implement deep learning models for recommendations.
- Integrate real-time recommendation updates.
- Deploy as a web API for production use.

## Contributors
- Bezalel Odiwuor (odiwuorvictory@gmail.com)
