import pandas as pd
import random

spotify_df = pd.read_csv("/Users/anjalinuggehalli/Desktop/ANNFinalProject/spotify/spotify_songs.csv")

weather_data_mapping = {
    "lightning": {
        "emotion": "intense",
        "danceability": (0.70, 0.95),
        "loudness": (-5.0, 1.0),
        "speechiness": (0.20, 0.60),
        "acousticness": (0.00, 0.20),
        "instrumentalness": (0.00, 0.40),
        "liveness": (0.40, 1.00),
        "valence": (0.70, 1.00),
        "tempo": (150, 240)
    },
    "rain": {
        "emotion": "melancholic",
        "danceability": (0.30, 0.65),
        "loudness": (-20.0, -5.0),
        "speechiness": (0.05, 0.40),
        "acousticness": (0.20, 0.80),
        "instrumentalness": (0.10, 0.70),
        "liveness": (0.10, 0.50),
        "valence": (0.00, 0.30),
        "tempo": (60, 130)
    },
    "snow": {
        "emotion": "cozy",
        "danceability": (0.20, 0.50),
        "loudness": (-30.0, -7.0),
        "speechiness": (0.00, 0.20),
        "acousticness": (0.60, 1.00),
        "instrumentalness": (0.40, 1.00),
        "liveness": (0.00, 0.30),
        "valence": (0.50, 0.75),
        "tempo": (50, 100)
    },
    "sandstorm": {
        "emotion": "eerie",
        "danceability": (0.60, 0.85),
        "loudness": (-10.0, 0.5),
        "speechiness": (0.30, 0.70),
        "acousticness": (0.00, 0.30),
        "instrumentalness": (0.10, 0.70),
        "liveness": (0.30, 0.80),
        "valence": (0.20, 0.50),
        "tempo": (120, 180)
    },
    "rime": {
        "emotion": "peaceful",
        "danceability": (0.20, 0.45),
        "loudness": (-35.0, -10.0),
        "speechiness": (0.00, 0.20),
        "acousticness": (0.70, 1.00),
        "instrumentalness": (0.30, 1.00),
        "liveness": (0.00, 0.30),
        "valence": (0.30, 0.60),
        "tempo": (0, 90)
    },
    "frost": {
        "emotion": "sharpness",
        "danceability": (0.40, 0.65),
        "loudness": (-15.0, -5.0),
        "speechiness": (0.10, 0.40),
        "acousticness": (0.10, 0.50),
        "instrumentalness": (0.10, 0.60),
        "liveness": (0.20, 0.60),
        "valence": (0.10, 0.40),
        "tempo": (100, 160)
    },
    "rainbow": {
        "emotion": "inspiring",
        "danceability": (0.60, 0.95),
        "loudness": (-10.0, 0.5),
        "speechiness": (0.05, 0.25),
        "acousticness": (0.30, 0.70),
        "instrumentalness": (0.10, 0.60),
        "liveness": (0.10, 0.50),
        "valence": (0.70, 1.00),
        "tempo": (100, 180)
    },
    "hail": {
        "emotion": "angry",
        "danceability": (0.50, 0.80),
        "loudness": (-12.0, 1.0),
        "speechiness": (0.20, 0.60),
        "acousticness": (0.00, 0.30),
        "instrumentalness": (0.00, 0.40),
        "liveness": (0.20, 0.70),
        "valence": (0.00, 0.30),
        "tempo": (130, 210)
    },
    "glaze": {
        "emotion": "elegant",
        "danceability": (0.40, 0.70),
        "loudness": (-12.0, -3.0),
        "speechiness": (0.05, 0.25),
        "acousticness": (0.30, 0.70),
        "instrumentalness": (0.20, 0.60),
        "liveness": (0.10, 0.40),
        "valence": (0.40, 0.70),
        "tempo": (90, 130)
    },
    "fogsmog": {
        "emotion": "edgy",
        "danceability": (0.40, 0.70),
        "loudness": (-15.0, -5.0),
        "speechiness": (0.20, 0.45),
        "acousticness": (0.10, 0.50),
        "instrumentalness": (0.10, 0.50),
        "liveness": (0.20, 0.60),
        "valence": (0.20, 0.50),
        "tempo": (100, 160)
    },
    "dew": {
        "emotion": "soft",
        "danceability": (0.30, 0.60),
        "loudness": (-20.0, -7.0),
        "speechiness": (0.05, 0.20),
        "acousticness": (0.40, 0.80),
        "instrumentalness": (0.20, 0.60),
        "liveness": (0.05, 0.30),
        "valence": (0.50, 0.75),
        "tempo": (60, 120)
    }
}


def get_song_for_emotion(emotion, weather_data_mapping, spotify_df):
    # Find the weather condition that matches the emotion
    weather_entry = next((v for k, v in weather_data_mapping.items() if v["emotion"] == emotion), None)
    if not weather_entry:
        return "No matching weather entry found for this emotion."

    # Filter songs that match all the intervals for that emotion
    mask = (
        (spotify_df["danceability"].between(*weather_entry["danceability"])) &
        (spotify_df["loudness"].between(*weather_entry["loudness"])) &
        (spotify_df["speechiness"].between(*weather_entry["speechiness"])) &
        (spotify_df["acousticness"].between(*weather_entry["acousticness"])) &
        (spotify_df["instrumentalness"].between(*weather_entry["instrumentalness"])) &
        (spotify_df["liveness"].between(*weather_entry["liveness"])) &
        (spotify_df["valence"].between(*weather_entry["valence"])) &
        (spotify_df["tempo"].between(*weather_entry["tempo"]))
    )

    matching_songs = spotify_df[mask]

    if matching_songs.empty:
        return "No songs match the criteria for this emotion."

    # Pick a random song
    song = matching_songs.sample(n=1).iloc[0]
    return {
        "track_name": song["track_name"],
        "artist": song["track_artist"],
        "valence": song["valence"],
        "tempo": song["tempo"],
        "danceability": song["danceability"],
        "preview_url": song.get("track_id", None),  # or add real URL if available
    }


