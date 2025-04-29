from WeatherCNN import predict_weather_and_emotion
from EmotionToRecipe import load_recipes, get_recipe_for_emotion
from EmotionToSpotify import get_song_for_emotion, weather_data_mapping, spotify_df


def full_pipeline(image_path):
    weather_label, emotion = predict_weather_and_emotion(image_path)
    song = get_song_for_emotion(emotion, weather_data_mapping, spotify_df)
    recipe = get_recipe_for_emotion(emotion, load_recipes())

    return (
        f"🌦️ Weather: {weather_label}\n"
        f"💬 Emotion: {emotion}\n\n"
        f"🎵 Song:\n  Title: {song['track_name']}\n  Artist: {song['artist']}\n\n"
        f"🍽️ Recipe:\n  Title: {recipe['title']}\n  Ingredients:\n" +
        "\n".join([f"    - {ingredient}" for ingredient in eval(recipe['ingredients'])])
    )

if __name__ == "__main__":
    image_path = "/Users/anjalinuggehalli/Desktop/ANNFinalProject/weather/lightning/1833.jpg"
    result = full_pipeline(image_path)
    print(result)


