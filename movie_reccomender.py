import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon = "🎬",
    layout="wide"
)

# custom css for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .time-suggestion {
        background-color: #e8f5e8;
        color: black;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all required data and models"""
    try:
        # load movies data
        movies = pd.read_csv("./movies.csv")
        
        # load artifacts
        artifacts_path = Path("artifacts")
        if artifacts_path.exists():
            # load model artifacts
            P_norm = np.load(artifacts_path / "mf_item_factors.npy")
            idx_to_item = pd.read_csv(artifacts_path / "mf_item_index.csv", index_col=0)
            idx_to_item = idx_to_item.iloc[:, 0] 
            
            # build lookup
            item_to_row = {int(idx_to_item.iloc[i]): i for i in range(len(idx_to_item))}
            
            return movies, P_norm, idx_to_item, item_to_row
        else:
            st.error("Artifacts folder not found! Please run your notebook first to generate the model artifacts.")
            return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def find_movie_ids_by_title(title_query, movies_df, k=10):
    """Find movies by title search"""
    q = title_query.strip().lower()
    hits = movies_df[movies_df['title'].str.lower().str.contains(q, na=False)]
    exact = movies_df[movies_df['title'].str.lower() == q]
    if not exact.empty:
        hits = pd.concat([exact, hits.loc[~hits.index.isin(exact.index)]], axis=0)
    return hits.head(k)[['movieId','title','genres']].reset_index(drop=True)

def similar_items_mf(seed_item_id, P_norm, idx_to_item, item_to_row, n=10):
    """Get similar items using matrix factorization"""
    row = item_to_row.get(int(seed_item_id))
    if row is None:
        raise ValueError(f"Item {seed_item_id} not in MF model.")

    # cosine similarity with every item
    sims = P_norm @ P_norm[row]  
    sims[row] = -np.inf  
    top = np.argpartition(-sims, kth=min(n, sims.size-1))[:n]
    top = top[np.argsort(-sims[top])]
    
    out = pd.DataFrame({
        'movieId': [int(idx_to_item.iloc[i]) for i in top],
        'score': sims[top]
    })
    return out

def get_time_suggestion(time_bucket):
    """Get a friendly time suggestion"""
    suggestions = {
        'Morning': "☀️ This would be an awesome morning watch!",
        'Afternoon': "🌤️ Take the afternoon easy and relax!",
        'Evening': "🌅 Ideal for unwinding in the evening after work or dinner!",
        'Night': "🌙 Perfect for late-night viewing and relaxation!"
    }
    return suggestions.get(time_bucket, "🎬 Enjoy your movie!")

def predict_best_watch_time(genres_str):
    """Predict the best time to watch a movie based on genre characteristics"""
    if not isinstance(genres_str, str) or not genres_str:
        return "Evening", 0.0
    
    # content-based time recommendations based on genre characteristics
    genre_time_mapping = {
        'Horror': 'Night',
        'Thriller': 'Evening', 
        'Action': 'Evening',
        'Crime': 'Evening',
        'Mystery': 'Evening',
        'Sci-Fi': 'Night',
        'Fantasy': 'Night',
        'Comedy': 'Afternoon',
        'Romance': 'Evening',
        'Animation': 'Afternoon',
        'Children': 'Morning',
        'Family': 'Afternoon',
        'Documentary': 'Afternoon',
        'Drama': 'Evening',
        'Musical': 'Afternoon',
        'Adventure': 'Evening',
        'War': 'Evening',
        'Western': 'Afternoon',
        'Film-Noir': 'Night',
        'IMAX': 'Evening'
    }
    
    genres = [g.strip() for g in genres_str.split('|')]
    time_votes = {'Morning': 0, 'Afternoon': 0, 'Evening': 0, 'Night': 0}
    
    for genre in genres:
        if genre in genre_time_mapping:
            time_votes[genre_time_mapping[genre]] += 1
    
    # if no genres match, default to Evening
    if sum(time_votes.values()) == 0:
        return "Evening", 0.0
    
    best_time = max(time_votes, key=time_votes.get)
    confidence = time_votes[best_time] / len(genres)
    
    return best_time, confidence

def main():
    # header
    st.markdown('<h1 class="main-header"> 🎥 Welcome to the Movie Recommendation System! </h1>', unsafe_allow_html=True)
    
    # load data
    movies, P_norm, idx_to_item, item_to_row = load_data()
    
    if movies is None:
        st.stop()
    
    # sidebar for input
    st.sidebar.header("⬇️ Movie You've Watched")
    
    # movie search
    movie_query = st.sidebar.text_input(
        "Search for a movie:",
        value=st.session_state.get('movie_query', ''),
        placeholder="e.g., Toy Story, Inception, The Matrix"
    )
    
    # number of recommendations
    num_recs = st.sidebar.slider("Number of recommendations:", 5, 15, 10)
    
    # main content
    if movie_query:
        try:
            # find movies matching the query
            movie_candidates = find_movie_ids_by_title(movie_query, movies, k=10)
            
            if movie_candidates.empty:
                st.warning(f"No movies found matching '{movie_query}'. Try a different search term.")
            else:
                # if multiple matches, let user choose
                if len(movie_candidates) > 1:
                    st.subheader("🔍 Multiple movies found. Please select one:")
                    
                    selected_idx = st.selectbox(
                        "Choose the movie you meant:",
                        range(len(movie_candidates)),
                        format_func=lambda x: f"{movie_candidates.iloc[x]['title']} ({movie_candidates.iloc[x]['genres']})"
                    )
                else:
                    selected_idx = 0
                
                selected_movie = movie_candidates.iloc[selected_idx]
                seed_id = int(selected_movie['movieId'])
                
                # show selected movie
                st.subheader("🎬 Selected Movie")
                st.markdown(f"""
                <div class="movie-card">
                    <div class="recommendation-title">{selected_movie['title']}</div>
                    <div>Genres: {selected_movie['genres']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # get recommendations
                with st.spinner("Finding similar movies..."):
                    # get base recommendations
                    base_recs = similar_items_mf(seed_id, P_norm, idx_to_item, item_to_row, n=50)
                    base_recs = base_recs.merge(movies[['movieId','title','genres']], on='movieId', how='left')
                    
                    # display top recommendations with watch time predictions
                    st.subheader(f"🎯 Top {num_recs} Recommendations")
                    
                    for i in range(min(num_recs, len(base_recs))):
                        rec = base_recs.iloc[i]
                        
                        # predict best watch time for this movie
                        best_time, time_score = predict_best_watch_time(rec['genres'])
                        
                        col1, col2, col3 = st.columns([4, 1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="movie-card">
                                <div class="recommendation-title">{rec['title']}</div>
                                <div>Genres: {rec['genres']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="score-badge">
                                Score: {rec['score']:.3f}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="time-suggestion">
                                <strong>Best Time:</strong><br>
                                {best_time}<br>
                                {get_time_suggestion(best_time)}
                            </div>
                            """, unsafe_allow_html=True)
                
                # show explanation
                st.subheader("ℹ️ How This Works")
                st.info("""
                **Recommendation Process:**
                1. **Collaborative Filtering**: Finds movies similar to your selected movie based on how users rated them
                2. **Time Prediction**: Uses genre characteristics to predict the best viewing time
                3. **Personalized Scoring**: Uses similarity scores to rank recommendations
                
                **Time Predictions Based on Genre Characteristics:**
                - **Morning**: Comedy, Animation, Family-friendly content
                - **Afternoon**: Comedy, Animation, Children, Family, Documentary, Musical
                - **Evening**: Action, Adventure, Drama, Romance, Thriller, Crime, Mystery
                - **Night**: Horror, Sci-Fi, Fantasy, Film-Noir
                
                Based on common viewing preferences and genre characteristics!
                """)
        
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            st.info("Make sure you've run your notebook to generate the model artifacts!")
    
    else:
        # welcome message
        st.markdown("""
        This system uses **Matrix Factorization** and **Time Prediction** to find movies you'll love and tell you when to watch them.
        
        **How to use:**
        1. Enter a movie title you've watched in the sidebar
        2. Choose how many recommendations you want
        3. Get personalized recommendations with optimal watch times!
        
        **Features:**
        - 🎯 **Collaborative Filtering**: Based on how similar users rated movies
        - ⏰ **Time Prediction**: Tells you the best time to watch each recommended movie
        - 🎭 **Genre-Smart**: Uses genre patterns to predict optimal viewing times
        - 📊 **Scored**: Each recommendation has a confidence score
        """)
        
        # show some example searches
        st.subheader("💡 Try searching for:")
        examples = ["Toy Story", "The Matrix", "Inception", "Pulp Fiction", "The Dark Knight"]
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i]:
                if st.button(f"🔍 {example}", key=f"example_{i}"):
                    st.session_state.movie_query = example
                    st.rerun()

if __name__ == "__main__":
    main()