import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import difflib

app = Flask(__name__)
movie_user = pickle.load(open('movie_user.df', 'rb'))
dff =  pickle.load(open('dff.df', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    in_movie = [str(x) for x in request.form.values()][0]
    mo = movie_user.columns
    in_movie = difflib.get_close_matches(in_movie, mo)[0]

    correlations = movie_user.corrwith(movie_user[in_movie])
    recommendation = pd.DataFrame(correlations,columns=['Correlation'])
    recommendation.dropna(inplace=True)
    recommendation = recommendation.join(dff['tot_ratings'])


    recc = recommendation[recommendation['tot_ratings']>10].sort_values('Correlation',ascending=False).reset_index()
    output = recc.title[1]

    #output = round(prediction[0], 2)
    #output = round(2414214.245,2)

    return render_template('index.html', prediction_text=f'Input movie: {in_movie}\nRecommended movie: {output}')


if __name__ == "__main__":
    app.run(debug=True)