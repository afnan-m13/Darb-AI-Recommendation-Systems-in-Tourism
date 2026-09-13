# 🌍 AI Travel Recommendation System

An intelligent, persona-driven travel recommendation engine that generates contextually relevant, personalized itineraries across Saudi Arabia using semantic embeddings and multi-factor hybrid filtering.

---

## 💡 Overview & Core Idea

Traditional travel systems rely on basic static filters, often missing the psychological motivations and personality traits of a traveler. This project addresses that gap by capturing traveler personas and matching them with places and plans using Natural Language Processing.

The system converts user interests, traveler types, and plan traits into dense semantic vector embeddings using the `all-MiniLM-L6-v2` Sentence Transformer model. These vectors are then evaluated to deliver tailored recommendations through two distinct operational modes:

### 🧠 1. Content-Based Mode (Plan-to-Plan)
* Calculates cosine similarity between the semantic textual profile of travel plans.
* Filters and recommends top similar itineraries within the same city and traveler category.

### 👤 2. Hybrid Personalized Mode (User-to-Plan)
Combines multiple scoring signals into a weighted final score to recommend the most suitable itineraries for a specific user:
* **Semantic Match (50%):** Direct vector similarity between user profile text (persona, traveler category, interests) and plan traits.
* **Collaborative Boost (30%):** Alignment with travel plans previously liked by the user.
* **Rating Influence (10%):** Prioritizes highly rated itineraries.
* **Budget Score (10%):** Normalizes user budget preference against plan costs.

---

## 📊 Dataset Structure

The system processes three curated Excel datasets:

| Dataset | Records | Description |
| :--- | :--- | :--- |
| `places_dataset.xlsx` | 229 POIs | Contains place name, genre, personality traits, traveler category, budget (SAR), rating, and location. |
| `plans_dataset.xlsx` | 50 Plans | Pre-curated itineraries linking 3–4 places each, along with city location, price, category, and average rating. |
| `users_dataset.xlsx` | 150 Users | Profiles containing persona descriptions, traveler categories, interest lists, budget preferences, and historical plan IDs. |

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **User Interface:** Streamlit
* **Data Processing:** Pandas, NumPy
* **ML & Similarity Search:** `sentence-transformers` (`all-MiniLM-L6-v2`), `scikit-learn` (Cosine Similarity)

---

## 👥 Authors & Contributors

* **Authors:** Afnan Kamel, Aya Mohammed, Afrah Bashaddadah
* **Supervisor:** Dr. Passent Elkafrawy
