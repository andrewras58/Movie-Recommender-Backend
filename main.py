from fastapi import FastAPI
from contextlib import asynccontextmanager
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing as pre
import pickle
import os
import databases

top_n = 10

pickle_dir = "pickles"

models = {} # genre, overview, keyword
encodings = {} # genre, overview, keyword, popularity, review
model_weights = {
    'overview': 0.6,
    'popularity': 0.05,
    'review': 0.05,
    'keyword': 0.1,
    'genre': 0.2
}

database_url = os.getenv('DATABASE_URL')
database = databases.Database(database_url)

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(os.path.join(pickle_dir, 'my_models.pickle'), 'rb') as r:
        for key, value in pickle.load(r).items():
            models[key] = value

    with open(os.path.join(pickle_dir, 'my_encodings.pickle'), 'rb') as r:
        for key, value in pickle.load(r).items():
            encodings[key] = value

    await database.connect()

    yield

    await database.disconnect()
    models.clear()
    encodings.clear()
    model_weights.clear()

app = FastAPI(lifespan=lifespan)

'''
We need two endpoints for now:

    1) An endpoint to handle title searching
        input -> movie_title: string
        output -> movie_set: JSON object containing entries from the database based on movie_title (direct query)

    2) An endpoint to handle lookup after the AI has found the movie list
        input -> movie_id: int
        output -> movie_set: JSON object containing entries from the database based on IDs returned from the AI based on movie_id
'''

@app.get('/api/search-by-title')
async def search_by_title(title: str):
    query = '''
        SELECT id, title, release_date, poster_path FROM movies
        WHERE title LIKE :title
        LIMIT 10
    '''
    values = {'title': f'%{title}%'}
    results = await database.fetch_all(query=query, values=values)
    return results

@app.get('/api/resemblance-results')
async def resemblance_results(movie_id: int):
    # step 1: get the full movie data we are comparing against
    query = '''
        SELECT id, overview, genres, keywords FROM movies
        WHERE id = :movie_id
    '''
    values = {'movie_id': movie_id}
    movie = await database.fetch_one(query=query, values=values)
    genres = [movie['genres'].split(', ')]
    keywords = movie['keywords'].replace(', ', ' ')
    overview = movie['overview']

    overview_embedding = models['overview'].encode(overview, convert_to_tensor=True)
    keyword_encoding = models['keyword'].transform([keywords]).tocsc().astype(float)
    genre_matrix = models['genre'].transform(genres)

    overview_scores = util.pytorch_cos_sim(overview_embedding, encodings['overview'])
    keyword_scores = cosine_similarity(keyword_encoding, encodings['keyword'])
    genre_scores = cosine_similarity(genre_matrix, encodings['genre'])

    combined_score = np.array((
        model_weights['overview'] * overview_scores +
        model_weights['keyword'] * keyword_scores + 
        model_weights['popularity'] * encodings['popularity'] +
        model_weights['review'] * encodings['review'] + 
        model_weights['genre'] * genre_scores
    )[0])

    top_n_combined = np.array(np.argsort(-combined_score)[:top_n])

    # step 3: query the db using the order of top_n_combined
    query = '''
        SELECT id, title, release_date, poster_path, imdb_id FROM movies
        WHERE id IN :movie_ids
    '''
    values = {'movie_ids': tuple(top_n_combined)}
    movies = await database.fetch_all(query=query, values=values)

    return movies