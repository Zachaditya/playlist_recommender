from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


app = Flask(__name__, template_folder='templates')

# Load data ---------------------------------------------------
try:
    SE_data = pd.read_csv('models/search_engine_data.csv', sep=',')
    recommendation_data = pd.read_csv('models/recommendation_data.csv', sep=',')
    image_data = pd.read_csv('models/image_data.csv', sep=',')
except Exception as e:
    print(f"Error loading data: {e}")
    SE_data = pd.DataFrame()  # Create empty DataFrame as a fallback
    recommendation_data = pd.DataFrame()
    image_data = pd.DataFrame()

def load_popular_songs():
    df = pd.read_csv("models/image_data.csv", sep=",") 
    return df.to_dict(orient="records")  

def load_popular_artist():
    df = pd.read_csv("models/popular_artists.csv", sep=",") 
    return df.to_dict(orient="records")

def search_engine(query):
    song_names = SE_data['Name']

    inverted_index = defaultdict(list)
    for i, song in enumerate(song_names):
        for word in song.lower().split():
            inverted_index[word].append(i)

    query_words = query.lower().split()

    matched_indices = set()
    for word in query_words:
        if word in inverted_index:
            matched_indices.update(inverted_index[word])  

    if not matched_indices:
        return pd.DataFrame()

    results_df = SE_data.iloc[list(matched_indices)]

    results_df = results_df.sort_values(by='Popularity', ascending=False)

    return results_df


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/songs")
def get_popular_songs():
    songs = load_popular_songs()  
    return jsonify(songs) 

@app.route("/artists")
def get_popular_artists():
    artists = load_popular_artist()  
    return jsonify(artists)

@app.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        query = request.form['query'].strip()  
        results = search_engine(query)

        if results.empty:
            return render_template('index.html', message="No results found.")
        else:
            return render_template('index.html', results=results.to_dict('records'), query=query)

    return render_template('index.html')


@app.route('/main')
def main():
    return render_template('main.html')

    
if __name__ == '__main__':
    app.run(debug=True)

