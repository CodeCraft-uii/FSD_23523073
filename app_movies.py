import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Debugging - Cek keberadaan file movies.csv
st.title("Movie Recommender System - Debug Mode")
file_path = "movies.csv"
st.write("Cek keberadaan file 'movies.csv':", os.path.exists(file_path))
st.write("Current working directory:", os.getcwd())

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(file_path)
        st.write("Dataset berhasil dimuat! Jumlah baris:", data.shape[0], "dan kolom:", data.shape[1])
        return data
    except Exception as e:
        st.error("Error loading dataset: " + str(e))
        return pd.DataFrame()

movies_data = load_data()

if not movies_data.empty:
    st.write("Preview Dataset:")
    st.write(movies_data.head())

    # Preprocess data
    def combine_features(row):
        return row['genres'] + ' ' + row['keywords'] + ' ' + row['tagline'] + ' ' + row['cast'] + ' ' + row['director']

    required_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    missing_columns = [feature for feature in required_features if feature not in movies_data.columns]

    if missing_columns:
        st.error(f"Kolom yang diperlukan tidak ditemukan dalam dataset: {missing_columns}")
    else:
        for feature in required_features:
            movies_data[feature] = movies_data[feature].fillna('')

        movies_data['combined_features'] = movies_data.apply(combine_features, axis=1)
        st.write("Fitur Gabungan (Combined Features):")
        st.write(movies_data[['title', 'combined_features']].head())

        # Build similarity matrix
        try:
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(movies_data['combined_features'])
            similarity_score = cosine_similarity(count_matrix)
            st.write("Similarity matrix berhasil dibuat! Shape:", similarity_score.shape)
        except Exception as e:
            st.error("Error creating similarity matrix: " + str(e))
            similarity_score = []

        # Function to get movie recommendations
        def get_recommendations(movie_title):
            try:
                movie_index = movies_data[movies_data.title == movie_title].index[0]
                similar_movies = list(enumerate(similarity_score[movie_index]))
                sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
                
                recommended_movies = []
                for i, movie in enumerate(sorted_similar_movies[1:11]):  # Get top 10 recommendations
                    index = movie[0]
                    title_from_index = movies_data.iloc[index].title
                    recommended_movies.append((i+1, title_from_index))
                return recommended_movies
            except IndexError:
                return None
            except Exception as e:
                st.error("Error in recommendation function: " + str(e))
                return None

        # Streamlit UI
        st.title("Movie Recommender System")
        st.write("Find movies similar to your favorites!")

        # Input from user
        movie_list = movies_data['title'].dropna().unique()
        selected_movie = st.selectbox("Select a movie:", sorted(movie_list))

        if st.button("Recommend"):
            recommendations = get_recommendations(selected_movie)
            if recommendations:
                st.write(f"Movies recommended for you based on **{selected_movie}**:")
                for rank, movie in recommendations:
                    st.write(f"{rank}. {movie}")
            else:
                st.write("Sorry, the selected movie was not found in the dataset.")
else:
    st.error("Dataset tidak berhasil dimuat. Pastikan file 'movies.csv' tersedia di direktori yang sama dengan script.")
