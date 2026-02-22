import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration (Wide layout for a dashboard feel)
st.set_page_config(
    page_title="Career Matcher Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Advanced CSS: Hover animations, gradients, and modern card design
st.markdown("""
    <style>
    /* Gradient Text for Main Header */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    /* Animated Hover Cards */
    .job-card {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        border-left: 8px solid #2196F3;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        color: #1e1e1e;
        height: 100%;
    }
    .job-card:hover {
        transform: translateY(-5px);
        box-shadow: 5px 10px 20px rgba(0,0,0,0.1);
        border-left: 8px solid #4CAF50;
    }
    /* Dark Mode Support */
    [data-theme="dark"] .job-card {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        border: 1px solid #333;
        border-left: 8px solid #2196F3;
        color: #f1f1f1;
    }
    .job-title { margin: 0; font-size: 1.2em; font-weight: 700; }
    .match-score { color: #888; font-size: 0.9em; margin-top: 5px; font-weight: 500;}
    </style>
""", unsafe_allow_html=True)

# 3. Optimized Data Loading
@st.cache_data
def load_data():
    df = pickle.load(open('df.pkl', 'rb'))
    df['Title'] = df['Title'].str.title()
    df['Title'] = df['Title'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
    return df

@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Title'])
    return cosine_similarity(tfidf_matrix)

try:
    df = load_data()
    similarity = compute_similarity(df)
except FileNotFoundError:
    st.error("Error: Could not find 'df.pkl'.")
    st.stop()

# 4. Upgraded Recommendation Engine (Now returns scores)
def recommendation(title):
    idx = df[df['Title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11] # Top 10 matches
    
    # Return both the job title and the similarity score (converted to percentage)
    jobs = [(df.iloc[i[0]]['Title'], round(i[1] * 100, 1)) for i in scores]
    return jobs

# --- Main UI ---

st.markdown('<p class="gradient-text">AI Career Matcher Pro</p>', unsafe_allow_html=True)
st.markdown("*Find your next role with mathematical precision using TF-IDF NLP vectors.*")

# Top Metrics Row
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total Jobs in Database", f"{len(df):,}")
col_m2.metric("Algorithm", "Cosine Similarity")
col_m3.metric("System Status", "Optimized & Active", delta="100%", delta_color="normal")

st.divider()

# Search Section
col_search, col_btn = st.columns([4, 1])
with col_search:
    selected_title = st.selectbox(
        "Enter your current or desired role:", 
        options=df['Title'].unique(), 
        index=None, 
        placeholder="Search for roles (e.g., Software Engineer)..."
    )

with col_btn:
    st.write("") # Spacer to align button with input
    st.write("")
    trigger_search = st.button("Analyze Matches", type="primary", use_container_width=True)

# Results Section
if trigger_search:
    if not selected_title:
        st.warning("Please select a job role to begin the analysis.")
    else:
        with st.spinner("Calculating semantic vectors..."):
            results = recommendation(selected_title)
            
            # Convert results to dataframe for easy charting/downloading
            results_df = pd.DataFrame(results, columns=["Recommended Role", "Match Percentage"])
            results_df.index = results_df.index + 1
        
        st.success(f"Analysis complete! Found top 10 matches for **{selected_title}**.")
        
        # --- Interactive Tabs ---
        tab1, tab2, tab3 = st.tabs(["Grid View", "Analytics", "Raw Data"])
        
        with tab1:
            # Create a 2-column grid for the cards
            col1, col2 = st.columns(2)
            
            for i, (job, score) in enumerate(results):
                # Alternate between columns
                with col1 if i % 2 == 0 else col2:
                    # Normalize score for progress bar (0.0 to 1.0)
                    progress_val = min(score / 100.0, 1.0) 
                    
                    st.markdown(f"""
                    <div class="job-card">
                        <p class="job-title">{i+1}. {job}</p>
                        <p class="match-score">Match Accuracy: {score}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    # Native Streamlit progress bar right below the card
                    st.progress(progress_val)
                    st.write("") # slight spacing

        with tab2:
            st.subheader("Match Proximity Chart")
            # Bar chart visualizing the match percentages
            st.bar_chart(data=results_df.set_index("Recommended Role"), height=400)

        with tab3:
            st.subheader("Tabular Data")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button for CSV
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f'job_matches_{selected_title.replace(" ", "_")}.csv',
                mime='text/csv',
            )