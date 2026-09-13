import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# SAFE PARSER 
# ---------------------------------------------------------
def safe_parse_list(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or x.strip() == "":
        return []
    x = x.replace("[", "").replace("]", "")
    x = x.replace("'", "").replace('"', "")
    items = [item.strip() for item in x.split(",") if item.strip()]
    return items

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():

    places = pd.read_excel("places_dataset.xlsx")
    plans = pd.read_excel("plans_dataset.xlsx")
    users = pd.read_excel("users_dataset.xlsx")

    places.columns = places.columns.str.lower().str.strip()
    plans.columns = plans.columns.str.lower().str.strip()
    users.columns = users.columns.str.lower().str.strip()

    # RENAME PLACE COLUMNS
    place_info = places.rename(columns={
        "place_name": "place_name",
        "place_genre": "place_genre",
        "place_city": "city",
        "place_personality_trait": "place_trait",
        "place_traveler_category": "place_traveler",
        "place_budget_per_person_sar": "place_budget",
        "place_rating": "place_rating",
        "place_location": "place_location"
    })

    # EXPAND PLANS TABLE
    plan_long = pd.melt(
        plans,
        id_vars=[
            "plan_id", "plan_city", "plan_traveler_category",
            "plan_price_sar", "plan_name",
            "plan_genre", "plan_traits",
            "plan_avg_rating"
        ],
        value_vars=["place_1", "place_2", "place_3", "place_4"],
        var_name="place_order",
        value_name="place_id"
    )

    plan_long = plan_long.merge(place_info, on="place_id", how="left")

    # CREATE SEMANTIC TEXT
    plan_text = plan_long.groupby("plan_id").agg({
        "place_genre": lambda x: " ".join(x.dropna()),
        "place_trait": lambda x: " ".join(x.dropna()),
        "place_traveler": lambda x: " ".join(x.dropna()),
        "plan_traveler_category": "first",
        "plan_city": "first",
        "plan_name": "first",
        "plan_price_sar": "first",
        "plan_avg_rating": "first"
    }).reset_index()

    plan_text["text"] = (
        plan_text["place_genre"].fillna("") + " " +
        plan_text["place_trait"].fillna("") + " " +
        plan_text["place_traveler"].fillna("") + " " +
        plan_text["plan_city"].fillna("")
    )

    # USERS
    users["user_past_plans"] = users["user_past_plans"].apply(safe_parse_list)

    users["profile_text"] = (
        users["user_persona"].fillna("") + " " +
        users["user_traveler_category"].fillna("") + " " +
        users["user_interests"].fillna("")
    )

    return places, plans, users, plan_long, plan_text


places, plans, users, plan_long, plan_text = load_data()

# ---------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

plan_embeddings = model.encode(plan_text["text"].tolist(), show_progress_bar=True)
user_embeddings = model.encode(users["profile_text"].tolist(), show_progress_bar=True)


def get_plan_places(plan_id):
    rows = plan_long[plan_long["plan_id"] == plan_id]
    return rows["place_name"].dropna().tolist()

# ---------------------------------------------------------
# CONTENT-BASED RECOMMENDER
# ---------------------------------------------------------
def recommend_by_plan(plan_name):

    try:
        index = plan_text[plan_text["plan_name"] == plan_name].index[0]
    except:
        st.error("Plan not found.") 
        return pd.DataFrame()

    sim_scores = cosine_similarity([plan_embeddings[index]], plan_embeddings)[0]
    plan_text["similarity"] = sim_scores

    selected_category = plan_text.loc[index, "plan_traveler_category"]
    selected_city = plan_text.loc[index, "plan_city"]

    same_type_city = plan_text[
        (plan_text["plan_traveler_category"].str.lower() == selected_category.lower()) &
        (plan_text["plan_city"].str.lower() == selected_city.lower())
    ]

    return same_type_city.sort_values("similarity", ascending=False).head(6).iloc[1:]

# ---------------------------------------------------------
# HYBRID RECOMMENDER
# ---------------------------------------------------------
def recommend_for_user(user_name):

    try:
        user_index = users[users["user_name"] == user_name].index[0]
    except:
        st.error("User not found.")
        return pd.DataFrame()

    user_vec = user_embeddings[user_index]
    user_budget = users.loc[user_index, "user_budget_preference"]
    traveler_type = users.loc[user_index, "user_traveler_category"]
    liked = users.loc[user_index, "user_past_plans"]

    semantic = cosine_similarity([user_vec], plan_embeddings)[0]
    plan_text["semantic_match"] = semantic

    max_budget = max(plan_text["plan_price_sar"].max(), user_budget)
    plan_text["budget_score"] = 1 - abs(plan_text["plan_price_sar"] - user_budget) / max_budget

    if liked:
        liked_vecs = plan_embeddings[plan_text["plan_id"].isin(liked)]
        collab = cosine_similarity(plan_embeddings, liked_vecs).mean(axis=1)
    else:
        collab = np.zeros(len(plan_text))

    plan_text["collaborative_boost"] = collab

    plan_text["final_score"] = (
        0.5 * semantic +
        0.3 * collab +
        0.1 * (plan_text["plan_avg_rating"] / 5) +
        0.1 * plan_text["budget_score"]
    )

    if liked:
        liked_cities = plan_text[plan_text["plan_id"].isin(liked)]["plan_city"]
        user_city = liked_cities.mode()[0] if not liked_cities.empty else None
    else:
        user_city = None

    if user_city:
        eligible = plan_text[
            (plan_text["plan_traveler_category"].str.lower() == traveler_type.lower()) &
            (plan_text["plan_city"].str.lower() == user_city.lower())
        ]
    else:
        eligible = plan_text[
            plan_text["plan_traveler_category"].str.lower() == traveler_type.lower()
        ]

    return eligible.sort_values("final_score", ascending=False).head(5)

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Travel Recommender", layout="wide")
st.title("🌍 AI Travel Recommendation System")

mode = st.radio("Choose mode:", ["🧠 By Plan (Content-Based)", "👤 By User (Hybrid)"])

# ---------------------------------------------------------
# CONTENT-BASED MODE
# ---------------------------------------------------------
if mode == "🧠 By Plan (Content-Based)":

    selected = st.selectbox("Select a Plan:", plan_text["plan_name"])

    if st.button("✨ Show Similar Plans"):
        recs = recommend_by_plan(selected)

        if recs.empty:
            st.warning("No similar plans found in the same city.")
        else:
            st.success(f"Top recommendations similar to **{selected}**")

            cols = st.columns(5)
            for i, col in enumerate(cols[:len(recs)]):
                row = recs.iloc[i]
                with col:
                    st.markdown(f"### {row['plan_name']}")
                    st.caption(f"📍 {row['plan_city']}")
                    st.write(f"👥 {row['plan_traveler_category']}")
                    st.write(f"💰 {row['plan_price_sar']} SAR")
                    st.write(f"⭐ {row['plan_avg_rating']}")

                    st.write("🗺️ Included Places:")
                    for p in get_plan_places(row["plan_id"]):
                        st.write(f"• {p}")

                    st.divider()

# ---------------------------------------------------------
# USER MODE
# ---------------------------------------------------------
if mode == "👤 By User (Hybrid)":

    selected = st.selectbox("Select a User:", users["user_name"])

    if st.button("✨ Show Personalized Recommendations"):
        recs = recommend_for_user(selected)

        if recs.empty:
            st.warning("No recommendations found.for this user.")
        else:
            st.success(f"Recommended plans for **{selected}**")

            cols = st.columns(5)
            for i, col in enumerate(cols[:len(recs)]):
                row = recs.iloc[i]
                with col:
                    st.markdown(f"### {row['plan_name']}")
                    st.caption(f"📍 {row['plan_city']}")
                    st.write(f"👥 {row['plan_traveler_category']}")
                    st.write(f"💰 {row['plan_price_sar']} SAR")
                    st.write(f"⭐ {row['plan_avg_rating']}")

                    st.write("🗺️ Included Places:")
                    for p in get_plan_places(row["plan_id"]):
                        st.write(f"• {p}")

                    st.divider()
