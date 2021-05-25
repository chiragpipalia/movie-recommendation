import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__)
##movie_features = pickle.load(open('movie_features.pickle', 'rb'))
cosine_sim = pickle.load(open('cosine_sim1k.pickle', 'rb'))
movies = pickle.load(open('movies.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    in_movie = [str(x) for x in request.form.values()][0]
    mo = list(movies['newname'])
    in_movie = difflib.get_close_matches(in_movie, mo)[0]
    movie_idx = dict(zip(movies['newname'], list(movies.index)))
    #title = movie_finder(title_string)
    idx = movie_idx[in_movie]
    #cosine_sim = cosine_similarity(movie_features, movie_features)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    n_recommendations = 5
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    #output = f"Recommendations for {in_movie}:\n"
    recommendations = movies["newname"].iloc[similar_movies].tolist()
    print_in_movie = 'Because you watched '+ in_movie
    might_like = 'You might also like'
    return render_template('index.html', in_movie = print_in_movie, might_like = might_like, recomms = recommendations)
    ##return render_template('index.html', prediction_text=f'Input movie: {in_movie}\nRecommended movies:\n{output}')
  

if __name__ == "__main__":
    app.run(debug=True)