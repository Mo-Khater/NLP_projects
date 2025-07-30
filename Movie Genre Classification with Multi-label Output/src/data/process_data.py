def preproces(movies):
    import pandas as pd 
    movies['overview'] = movies['overview'].fillna('')
    movies['overview_title'] = movies['overview'] + ' ' + movies['title']
    return movies[['title', 'overview_title', 'genres', 'cast', 'crew']]

def clean_and_tokenize(text):
    import re 
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()         
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


