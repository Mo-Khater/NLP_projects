def load_data(tmdb_5000_credits='data/raw/tmdb_5000_credits.csv',tmdb_5000_movies='data/raw/tmdb_5000_movies.csv'):
    import pandas as pd
    import json
    credits = pd.read_csv(tmdb_5000_credits)
    movies = pd.read_csv(tmdb_5000_movies)
    movies = movies[['genres','id','overview']]
    credits = credits[['movie_id', 'title', 'cast', 'crew']]
    credits['cast'] = credits['cast'].apply(lambda x: [dic['name'] for dic in json.loads(x)][:10])
    credits['crew'] =  credits['crew'].apply(lambda x: [dic['name'] for dic in json.loads(x) if dic['job'] in ('Director','Producer','Writer')][:15])
    movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
    movies['genres'] = movies['genres'].apply(lambda x: [dic['name'] for dic in json.loads(x)])
    return movies