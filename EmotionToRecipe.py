import pandas as pd
import re
from WeatherCNN import emotion


# Example: Load your dataset
recipes_df = pd.read_csv("/Users/anjalinuggehalli/Applications/ANNFinalProject/recipes/RecipeNLG_dataset.csv") 


emotion_to_recipe_keywords = {
    "intense": [
        "spicy", "buffalo", "curry", "sriracha", "jerk", "chili", 
        "kung pao", "wasabi", "bold flavors", "hot wings", "black pepper"
    ],
    "melancholic": [
        "warm soup", "tomato soup", "grilled cheese", "risotto", 
        "mushroom", "lentil stew", "slow cooked", "tea cake", "comforting", "soft texture"
    ],
    "cozy": [
        "mac and cheese", "hot chocolate", "chicken pot pie", 
        "casserole", "pancakes", "banana bread", "oatmeal", "pumpkin spice", "chili", "cinnamon"
    ],
    "eerie": [
        "charcoal", "black garlic", "squid ink", "miso", 
        "blood orange", "pomegranate", "beetroot", "dark chocolate", "unusual", "mystery"
    ],
    "peaceful": [
        "herbal tea", "green tea noodles", "cucumber salad", 
        "lemon balm", "miso soup", "rice porridge", "tofu", "zen", "light", "gentle"
    ],
    "sharpness": [
        "lemon tart", "pickled vegetables", "kimchi", 
        "sharp cheddar", "vinegar", "mustard glaze", "horseradish", "tangy", "zesty"
    ],
    "inspiring": [
        "rainbow salad", "smoothie bowl", "poke bowl", 
        "avocado toast", "colorful veggies", "fruit tart", "fusion", "vibrant", "creative"
    ],
    "angry": [
        "fiery", "spicy ramen", "jalapeño poppers", "hot wings", 
        "diablo sauce", "cajun", "red curry", "devil’s food cake", "pepper", "bold spice"
    ],
    "elegant": [
        "coq au vin", "truffle pasta", "soufflé", "tartare", 
        "champagne vinaigrette", "caprese", "brie", "pistachio crusted", "plated dessert", "delicate"
    ],
    "edgy": [
        "fusion tacos", "kimchi fries", "loaded nachos", 
        "street food", "bacon jam", "brioche bun", "unexpected combos", "cheddar jalapeño", "crunch"
    ],
    "soft": [
        "vanilla pudding", "egg custard", "cloud cake", 
        "milk bread", "buttermilk pancakes", "yogurt parfait", "steamed buns", "fluffy", "gentle sweetness"
    ]
}


def find_matching_recipes(emotion, recipes_df, top_n=1):
    keywords = emotion_to_recipe_keywords.get(emotion, [])
    if not keywords:
        return []

    # Combine keywords into regex pattern
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)

    # Filter recipes matching keywords in title or ingredients
    matches = recipes_df[
        recipes_df["title"].str.contains(pattern, na=False) |
        recipes_df["ingredients"].str.contains(pattern, na=False)
    ]

    # If there are fewer matches than top_n, just return all
    if len(matches) == 0:
        return []
    if len(matches) < top_n:
        return matches

    # Randomly sample without replacement
    return matches.sample(n=top_n, random_state=None)





# Let's say your predicted emotion is "cozy"
# emotion = "cozy"
matching_recipes = find_matching_recipes(emotion, recipes_df)

# View results
print(matching_recipes[["title", "ingredients"]])

