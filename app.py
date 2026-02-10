import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle

st.set_page_config(
    page_title="Dashboard Videojuegos",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Dashboard de Videojuegos Populares")

def convert_numbers(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        try:
            if x.endswith("K"):
                return float(x[:-1]) * 1_000
            elif x.endswith("M"):
                return float(x[:-1]) * 1_000_000
            else:
                return float(x)
        except:
            return np.nan
    return x


@st.cache_data
def load_data():
    df = pd.read_csv("backloggd_games.csv")
    return df


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model, scaler, model_name = pickle.load(f)
    return model, scaler, model_name

df = load_data()
df = df.dropna(how="all")

df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["year"] = df["Release_Date"].dt.year

st.sidebar.header("Filtros")

if "year" in df.columns:
    df = df.dropna(subset=["year"])

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    year_range = st.sidebar.slider(
        "Selecciona rango de años",
        min_year,
        max_year,
        (min_year, max_year)
    )

    df = df[
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1])
    ]

if "Genres" in df.columns:
    df["Genres"] = df["Genres"].fillna("")
    all_genres = sorted(
        set(
            genre.strip()
            for sublist in df["Genres"].str.split(",")
            for genre in sublist
            if genre
        )
    )

    selected_genres = st.sidebar.multiselect(
        "Selecciona género",
        options=all_genres,
        default=all_genres[:5]
    )

    if selected_genres:
        df = df[
            df["Genres"].apply(
                lambda x: any(g in x for g in selected_genres)
            )
        ]

st.subheader("🎯 Juegos por género")

if "Genres" in df.columns:
    exploded = df.assign(
        Genres=df["Genres"].str.split(",")
    ).explode("Genres")

    exploded["Genres"] = exploded["Genres"].str.strip()

    genre_count = (
        exploded["Genres"]
        .value_counts()
        .reset_index()
    )

    genre_count.columns = ["Genres", "count"]

    fig1 = px.bar(
        genre_count.head(10),
        x="Genres",
        y="count",
        title="Top 10 géneros más populares"
    )

    st.plotly_chart(fig1, use_container_width=True)


st.subheader("📈 Evolución de juegos por año")

if "year" in df.columns:
    games_per_year = (
        df.groupby("year")
        .size()
        .reset_index(name="count")
        .sort_values("year")
    )

    fig2 = px.line(
        games_per_year,
        x="year",
        y="count",
        markers=True,
        title="Cantidad de juegos lanzados por año"
    )

    st.plotly_chart(fig2, use_container_width=True)


st.subheader("⭐ Rating vs Popularidad")

if "Rating" in df.columns and "Plays" in df.columns:

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Plays"] = df["Plays"].apply(convert_numbers)

    fig3 = px.scatter(
        df,
        x="Rating",
        y="Plays",
        title="Relación entre Rating y Popularidad (Plays)",
        opacity=0.6,
        hover_data=["Title"]
    )

    st.plotly_chart(fig3, use_container_width=True)

st.subheader("🎯 Simulador de Reviews (Predicción ML)")

model, scaler, model_name = load_model()

st.write(f"Modelo activo: **{model_name}**")

col1, col2, col3 = st.columns(3)

with col1:
    plays = st.slider("Plays", 0, 1_000_000, 50_000, step=1000)

with col2:
    playing = st.slider("Playing", 0, 200_000, 5_000, step=500)

with col3:
    wishlist = st.slider("Wishlist", 0, 500_000, 10_000, step=1000)

engagement = playing / (plays + 1)

X_input = np.array([[playing, wishlist, plays, engagement]])

if model_name == "Linear Regression":
    X_input = scaler.transform(X_input)

prediction = model.predict(X_input)[0]

st.metric("📈 Reviews estimadas", f"{int(prediction):,}")


st.subheader("📋 Datos del dataset")
st.dataframe(df.head(50))
