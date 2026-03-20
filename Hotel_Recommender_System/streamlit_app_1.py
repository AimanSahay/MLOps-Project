import streamlit as st
import pandas as pd
import pickle

# IMPORTANT: Import the class (fixes pickle issue)
from recommender_1 import CFRecommender

st.set_page_config(page_title="Hotel Recommender", layout="centered")


# -------------------------------
# Load Data & Model
# -------------------------------
@st.cache_resource
def load_model():
    with open('recommender_1.pkl', 'rb') as f:
        recommender, cf_preds_df, interactions_df = pickle.load(f)
    return recommender, cf_preds_df, interactions_df


@st.cache_data
def load_data():
    return pd.read_csv('hotels.csv')


recommender, cf_preds_df, interactions_df = load_model()
hotel_df = load_data()


# -------------------------------
# App UI
# -------------------------------
def main():

    # Header
    st.title("🏨 Hotel Recommendation System")
    st.markdown(
        "Get personalized hotel recommendations using Collaborative Filtering."
    )

    # Sidebar
    st.sidebar.header("User Selection")

    user_list = sorted(hotel_df['userCode'].unique())
    selected_user = st.sidebar.selectbox("Select User Code", user_list)

    topn = st.sidebar.slider("Number of Recommendations", 3, 20, 5)

    st.sidebar.markdown("---")

    # Button
    if st.sidebar.button("🔍 Get Recommendations"):

        try:
            # Get seen items
            seen_items = interactions_df[
                interactions_df['userCode'] == selected_user
            ]['name_encoded'].tolist()

            # Get recommendations
            recommendations = recommender.recommend_items(
                user_id=selected_user,
                items_to_ignore=seen_items,
                topn=topn
            )

            if recommendations.empty:
                st.warning("No recommendations found.")
            else:
                st.success(f"Top {topn} recommendations for User {selected_user}")
                st.dataframe(recommendations, use_container_width=True)

        except KeyError:
            st.error("This user does not have enough interaction data.")


    # Footer
    st.markdown("---")
    st.markdown(
        "Built using Collaborative Filtering (SVD) • Streamlit App"
    )


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()