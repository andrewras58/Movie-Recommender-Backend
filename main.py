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
from scipy.sparse import csr_matrix
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

top_n = 40
display_limit = 10

pickle_dir = "pickles"

models = {} # genre, overview, keyword
encodings = {} # genre, overview, keyword, popularity, review
weights = {}

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

    with open(os.path.join(pickle_dir, 'my_weights.pickle'), 'rb') as r:
        for key, value in pickle.load(r).items():
            weights[key] = value

    await database.connect()

    yield

    await database.disconnect()
    models.clear()
    encodings.clear()
    weights.clear()

app = FastAPI(lifespan=lifespan)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        SELECT id, title, release_date, poster FROM movies
        WHERE title LIKE :title
        ORDER BY num_ratings DESC
        LIMIT 15
    '''
    values = {'title': f'%{title}%'}
    results = await database.fetch_all(query=query, values=values)
    return results

@app.get('/api/resemblance-results')
async def resemblance_results(movie_id: int):
    # step 1: get the full movie data we are comparing against
    query = '''
        SELECT id, overview, genres, keywords, cast, director, series FROM movies
        WHERE id = :movie_id
    '''
    values = {'movie_id': movie_id}
    movie = await database.fetch_one(query=query, values=values)

    genres = movie['genres']
    keywords = movie['keywords']
    overview = movie['overview']
    actors = movie['cast']
    directors = movie['director']
    series = movie['series']

    overview_embedding = models['overview'].encode(overview, convert_to_tensor=True)
    keyword_encoding = models['keyword'].transform([keywords.replace(', ', ' ')]).tocsc().astype(float)
    genre_matrix = csr_matrix(models['genre'].transform([genres.split(', ')]))
    actor_matrix = csr_matrix(models['actors'].transform([actors.split(', ')]))
    director_matrix = csr_matrix(models['director'].transform([directors.split(', ')]))

    overview_scores = util.pytorch_cos_sim(overview_embedding, encodings['overview'])
    keyword_scores = cosine_similarity(keyword_encoding, encodings['keyword']).flatten()
    genre_scores = cosine_similarity(genre_matrix, encodings['genre'])
    actor_scores = cosine_similarity(actor_matrix, encodings['actors'])
    director_scores = cosine_similarity(director_matrix, encodings['director'])

    combined_score = np.array((
        weights['overview'] * overview_scores.cpu() +
        weights['keyword'] * keyword_scores + 
        weights['popularity'] * encodings['popularity'] +
        weights['review'] * encodings['review'] + 
        weights['genre'] * genre_scores +
        weights['actors'] * actor_scores +
        weights['directors'] * director_scores
    )[0])

    top_n_combined = np.array(np.argsort(-combined_score)[:top_n])

    # step 3: query the db using the order of top_n_combined

    # NEXT TIME: change this query to work with the updated DB: exclude movies from the same collection as the provided ID
    query = f'''
        SELECT id, title, release_date, poster, imdb_id FROM movies
        WHERE id IN :movie_ids AND CASE
            WHEN (series != :series OR series = 0) AND id != :movie_id THEN 1
            ELSE 0
        END
        ORDER BY FIELD(id, {','.join(map(str, top_n_combined))})
        LIMIT :display_limit
    ''' 
    # gonna be honest here, no idea why f strings work here and not just putting it into values
    values = {'movie_ids': tuple(top_n_combined), 'series': series, 'display_limit': display_limit, 'movie_id': movie_id}
    movies = await database.fetch_all(query=query, values=values)

    return movies