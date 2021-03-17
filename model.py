import pandas as pd
import pickle

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
movies['title'] = movies.apply(lambda x: x['title'][:-7].lower(), axis = 1)

data = ratings.merge(movies, how = 'left', on= 'movieId')


dff = data.groupby(['title']).agg(
    {
         'rating':'mean',    # Sum duration per group
         'userId': "count",  # get the count of networks
         
    }
)
dff.rename(columns = {'rating':'avg_rating', 'userId': 'tot_ratings'}, inplace = True)
pickle.dump(dff, open( "dff.df", "wb" ))

movie_user = data.pivot_table(index='userId',columns='title',values='rating')

pickle.dump(movie_user, open( "movie_user.df", "wb" ))

correlations = movie_user.corrwith(movie_user['toy story'])
recommendation = pd.DataFrame(correlations,columns=['Correlation'])
recommendation.dropna(inplace=True)
recommendation = recommendation.join(dff['tot_ratings'])


recc = recommendation[recommendation['tot_ratings']>10].sort_values('Correlation',ascending=False).reset_index()
print(recc.head(10))

