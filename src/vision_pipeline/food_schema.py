"""
Built-in food nutrition lookup table.

Values are approximate macros per 100g:
- kcal
- protein (g)
- fat (g)
- carbs (g)
"""

FOOD_NUTRITION = {
    # Proteins
    "chicken breast": {"kcal": 165, "protein": 31.0, "fat": 3.6, "carbs": 0.0},
    "turkey breast": {"kcal": 135, "protein": 29.0, "fat": 1.0, "carbs": 0.0},
    "beef steak": {"kcal": 250, "protein": 26.0, "fat": 15.0, "carbs": 0.0},
    "pork loin": {"kcal": 242, "protein": 27.0, "fat": 14.0, "carbs": 0.0},
    "salmon": {"kcal": 208, "protein": 20.0, "fat": 13.0, "carbs": 0.0},
    "tuna canned": {"kcal": 132, "protein": 29.0, "fat": 1.0, "carbs": 0.0},
    "cod": {"kcal": 82, "protein": 18.0, "fat": 0.7, "carbs": 0.0},
    "shrimp": {"kcal": 99, "protein": 24.0, "fat": 0.3, "carbs": 0.2},
    "egg": {"kcal": 155, "protein": 13.0, "fat": 11.0, "carbs": 1.1},
    "egg white": {"kcal": 52, "protein": 11.0, "fat": 0.2, "carbs": 0.7},

    # Dairy
    "whole milk": {"kcal": 61, "protein": 3.2, "fat": 3.3, "carbs": 4.8},
    "skim milk": {"kcal": 35, "protein": 3.4, "fat": 0.1, "carbs": 5.0},
    "yogurt plain": {"kcal": 60, "protein": 5.0, "fat": 3.0, "carbs": 4.5},
    "cheddar cheese": {"kcal": 403, "protein": 25.0, "fat": 33.0, "carbs": 1.3},

    # Grains & starches (cooked)
    "rice cooked": {"kcal": 130, "protein": 2.7, "fat": 0.3, "carbs": 28.0},
    "brown rice cooked": {"kcal": 123, "protein": 2.7, "fat": 1.0, "carbs": 25.6},
    "pasta cooked": {"kcal": 150, "protein": 5.0, "fat": 1.0, "carbs": 30.0},
    "buckwheat cooked": {"kcal": 101, "protein": 3.4, "fat": 1.0, "carbs": 21.0},
    "oatmeal cooked": {"kcal": 71, "protein": 2.5, "fat": 1.5, "carbs": 12.0},
    "bread white": {"kcal": 265, "protein": 9.0, "fat": 3.2, "carbs": 49.0},
    "bread wholegrain": {"kcal": 247, "protein": 13.0, "fat": 4.2, "carbs": 41.0},

    # Vegetables
    "potato boiled": {"kcal": 87, "protein": 1.9, "fat": 0.1, "carbs": 20.0},
    "sweet potato baked": {"kcal": 90, "protein": 2.0, "fat": 0.2, "carbs": 21.0},
    "broccoli": {"kcal": 35, "protein": 2.4, "fat": 0.4, "carbs": 7.0},
    "carrot": {"kcal": 41, "protein": 0.9, "fat": 0.2, "carbs": 10.0},
    "tomato": {"kcal": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9},
    "cucumber": {"kcal": 15, "protein": 0.7, "fat": 0.1, "carbs": 3.6},
    "lettuce": {"kcal": 15, "protein": 1.4, "fat": 0.2, "carbs": 2.9},

    # Fruits
    "banana": {"kcal": 89, "protein": 1.1, "fat": 0.3, "carbs": 23.0},
    "apple": {"kcal": 52, "protein": 0.3, "fat": 0.2, "carbs": 14.0},
    "orange": {"kcal": 47, "protein": 0.9, "fat": 0.1, "carbs": 12.0},
    "strawberry": {"kcal": 33, "protein": 0.7, "fat": 0.3, "carbs": 8.0},
    "avocado": {"kcal": 160, "protein": 2.0, "fat": 15.0, "carbs": 9.0},

    # Fats & nuts
    "olive oil": {"kcal": 884, "protein": 0.0, "fat": 100.0, "carbs": 0.0},
    "butter": {"kcal": 717, "protein": 0.9, "fat": 81.0, "carbs": 0.1},
    "almonds": {"kcal": 579, "protein": 21.0, "fat": 50.0, "carbs": 22.0},
}
