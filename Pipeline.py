from WeatherCNN import predict_weather_and_emotion
from EmotionToRecipe import load_recipes, get_recipe_for_emotion
from EmotionToSpotify import get_song_for_emotion, weather_data_mapping, spotify_df


def full_pipeline(image_path):
    weather_label, emotion = predict_weather_and_emotion(image_path)
    song = get_song_for_emotion(emotion)
    recipe = get_recipe_for_emotion(emotion)

    return {
        "weather": weather_label,
        "emotion": emotion,
        "song": song,
        "recipe": recipe
    }

if __name__ == "__main__":
    image_path = "/Users/anjalinuggehalli/.Trash/ANNFinalProject/weather/dew/2208.jpg"
    result = full_pipeline(image_path)
    print(result)


