# Climate Cuisine: CNN-Based Music & Recipe Recommender

This project uses a Convolutional Neural Network (CNN) to classify weather conditions from images, maps them to corresponding emotions, and recommends a fitting song and recipe. It combines computer vision, emotion modeling, and content-based recommendation into a seamless pipeline.

---

## Project Structure

ANNFinalProject/
├── weather/                      # Weather image dataset (not pushed)
├── spotify/
│   └── spotify_songs.csv        # Songs dataset (not pushed)
├── recipes/
│   └── RecipeNLG_dataset.csv    # Recipe dataset (not pushed)
├── WeatherCNN.py                # CNN training code for weather prediction
├── EmotionToRecipe.py           # Recommends recipes based on emotion
├── EmotionToSpotify.py          # Recommends songs based on emotion
├── Pipeline.py                  # Full weather → emotion → recommendation pipeline
├── .gitignore
└── README.md


---


---

## Methodology

1. **Weather Classification:** A custom CNN model is trained using labeled weather images to classify input images into weather types like "lightning," "fog," or "rain."
2. **Emotion Mapping:** Each weather label is mapped to a unique emotion (e.g., "fogsmog" → "edgy", "lightning" → "intense").
3. **Recommendation Engine:** Given the predicted emotion, the system recommends:
   - A song from the Spotify dataset based on mood tags and audio features.
   - A recipe from RecipeNLG dataset based on mood and ingredient profile.

---

## Figures and How They Were Created

### `figures/Example1.png`  
> Full output of the model pipeline for a "lightning" image input.  
- **Weather classification:** `predict_weather_and_emotion()` in [`WeatherCNN.py`](./WeatherCNN.py)  
- **Recommendations:** `full_pipeline()` in [`Pipeline.py`](./Pipeline.py)  

### `figures/Example2.png`  
> Another result example with different input weather image.  
- **Same pipeline** and function references as above.

### `figures/WorkPipeline.png`  
> High-level overview of the system architecture: image → weather → emotion → music/recipe  
- **Created manually** to document the complete pipeline  

---

## Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com/anjalinugg4/ANNFinalProject.git
cd ANNFinalProject
