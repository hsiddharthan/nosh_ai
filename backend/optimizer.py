import itertools
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
import pulp

# SOURCE: https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels
FDA_GUIDELINES= {
    "macronutrients": {
        "calories": 2000,
        "protein": 50,
        "total_fat": 78,            
        "saturated_fat": 20,
        "cholesterol": 300,
        "total_carbohydrate": 275,
        "dietary_fiber": 28,
        "added_sugars": 50
    },
    "vitamins": {
        "choline": 550,
        "vitamin_a": 900,
        "vitamin_b6": 1.7,
        "vitamin_b12": 2.4,
        "vitamin_c": 90,
        "vitamin_d": 20,
        "vitamin_e": 15,
        "vitamin_k": 120
    },
    "minerals": {
        "calcium": 1300,
        "chloride": 2300,
        "chromium": 35,
        "copper": 0.9,
        "iodine": 150,
        "iron": 18,
        "magnesium": 420,
        "manganese": 2.3,
        "molybdenum": 45,
        "phosphorus": 1250,
        "potassium": 4700,
        "selenium": 55,
        "sodium": 2300,
        "zinc": 11
    }
}

DEF_TARGET_MACRO_RATIO = {k: v / (FDA_GUIDELINES["macronutrients"]["protein"] + FDA_GUIDELINES["macronutrients"]["total_fat"] + FDA_GUIDELINES["macronutrients"]["total_carbohydrate"]) for k, v in {"protein": FDA_GUIDELINES["macronutrients"]["protein"], "fat": FDA_GUIDELINES["macronutrients"]["total_fat"], "carbs": FDA_GUIDELINES["macronutrients"]["total_carbohydrate"]}.items()}


# Give higher weight to more important micros (example)
MICRO_WEIGHTS = {
    # core nutrients (higher)
    "iron": 2.0,
    "calcium": 2.0,
    "vitamin_d": 2.0,
    "vitamin_a": 2.0,
    "vitamin_c": 1.5,
    "vitamin_b12": 1.5,
    "vitamin_c": 1.5,
    "potassium": 1.8,
    "dietary_fiber": 2.0,
    "sodium": 2.0,
    "zinc": 1.5,
    # others default to 1.0 (or lower if you want)
}

DEFAULT_MICRO_WEIGHT = 1.0




# ------------------------------
# Utility Functions
# ------------------------------
def normalize(value, inverse=False):
    if isinstance(value, list):
        value = np.mean(value)
    if inverse:
        return 1 - min(1, max(0, value / 10))
    return min(1, max(0, value / 10))


def avg_cost(meals):
    return np.mean([m.get("cost", 0) for m in meals])


def avg_cook_time(meals):
    return np.mean([m.get("time", 0) for m in meals])

def calculate_macros(recipe):
    macros = recipe["nutrition"]["macronutrients"]
    calories = macros.get("calories", 0)
    if calories == 0:
        return {"protein": 0, "fat": 0, "carbs": 0}
    protein_cal = macros.get("protein",0) * 4
    fat_cal = macros.get("total_fat",0) * 9
    carb_cal = macros.get("total_carbohydrate",0) * 4
    total = protein_cal + fat_cal + carb_cal
    return {
        "protein": protein_cal / total if total else 0,
        "fat": fat_cal / total if total else 0,
        "carbs": carb_cal / total if total else 0,
    }


def match_macros(user_goal, meals):
    ratios = [calculate_macros(m) for m in meals]
    avg_ratio = {
        "protein": np.mean([r["protein"] for r in ratios]),
        "fat": np.mean([r["fat"] for r in ratios]),
        "carbs": np.mean([r["carbs"] for r in ratios]),
    }
    diff = sum(abs(user_goal[k] - avg_ratio[k]) for k in user_goal) / 3
    return 1 - diff

"""
TODO: Diversity should be based on actual cuisine types or ingredients
Rn there is a transformer that tests similarity by name
Should pull most important ingredients ie meat, spice profiles, etc. with Gemini
"""
def diversity_measure(candidate_meals, diversity_rule):
    if not diversity_rule:
        return 1.0
    cuisines = [m.get("cuisine", "unknown") for m in candidate_meals]
    return len(set(cuisines)) / len(cuisines)


def leftover_utilization(candidate_meals, prioritize_existing):
    return 1.0 if prioritize_existing else 0.5


def cuisine_match(preferences, candidate_meals):
    count = sum(m.get("cuisine", "unknown") in preferences for m in candidate_meals)
    return count / len(candidate_meals)


def evaluate_macros_and_micros(
    meals,
    user_prefs=None,
    target_macro_ratio=None,
    macro_primary_weight=0.85,
    macro_weight=0.65,
    micro_weight=0.35,
    verbose=False
):
    """
    Evaluate overall nutrition score from both macro and micro perspectives.
    Incorporates user preferences for specific nutrients.
    """
    if target_macro_ratio is None and user_prefs:
        target_macro_ratio = user_prefs.get("macro_goal_ratio", DEF_TARGET_MACRO_RATIO)
    elif target_macro_ratio is None:
        target_macro_ratio = DEF_TARGET_MACRO_RATIO

    nutrient_priorities = (user_prefs or {}).get("nutrient_priorities", {})

    # ---------- Aggregate nutrients ----------
    total = {}
    for m in meals:
        n = m.get("nutrition", {})
        for group in ("macronutrients", "vitamins", "minerals"):
            for k, v in n.get(group, {}).items():
                total[k] = total.get(k, 0) + v

    # ---------- MACROS ----------
    prot_g = total.get("protein", 0)
    fat_g = total.get("total_fat", 0)
    carb_g = total.get("total_carbohydrate", 0)
    cal_sum = total.get("calories", prot_g * 4 + fat_g * 9 + carb_g * 4)

    prot_cal, fat_cal, carb_cal = prot_g * 4, fat_g * 9, carb_g * 4
    denom = max(cal_sum, prot_cal + fat_cal + carb_cal, 1e-9)
    macro_ratio = {
        "protein": prot_cal / denom,
        "fat": fat_cal / denom,
        "carbs": carb_cal / denom
    }

    # Ratio score (difference from target)
    diff = sum(abs(macro_ratio[k] - target_macro_ratio.get(k, 0)) for k in target_macro_ratio)
    ratio_score = max(0.0, 1.0 - diff / 2.0)

    # Absolute component
    protein_ok = prot_g >= FDA_GUIDELINES["macronutrients"]["protein"]
    sat_fat_g = total.get("saturated_fat", 0)
    sat_fat_ok = sat_fat_g <= FDA_GUIDELINES["macronutrients"]["saturated_fat"]

    protein_bonus = 1.0 if protein_ok else prot_g / max(1.0, FDA_GUIDELINES["macronutrients"]["protein"])
    sat_fat_penalty = 1.0 if sat_fat_ok else max(0.0, 1.0 - (sat_fat_g - FDA_GUIDELINES["macronutrients"]["saturated_fat"]) / max(1.0, FDA_GUIDELINES["macronutrients"]["saturated_fat"]))

    absolute_component = min(protein_bonus, sat_fat_penalty)
    macro_score = macro_primary_weight * ratio_score + (1 - macro_primary_weight) * absolute_component
    macro_score = float(np.clip(macro_score, 0.0, 1.0))

    # ---------- MICROS ----------
    micro_scores = []
    micro_weights_list = []
    for __, nutrients in FDA_GUIDELINES.items():
        for nutrient, guideline in nutrients.items():
            if nutrient in ("protein", "total_fat", "total_carbohydrate", "calories", "saturated_fat"):
                continue

            actual = total.get(nutrient, 0)
            if guideline <= 0:
                continue

            ratio = actual / guideline
            if 0.9 <= ratio <= 1.1:
                nutrient_score = 1.0
            else:
                z = (ratio - 1.0) / 0.5
                nutrient_score = float(np.exp(-abs(z)))

            # Base weight
            base_w = MICRO_WEIGHTS.get(nutrient, DEFAULT_MICRO_WEIGHT)
            # User-specific multiplier (default 1.0)
            user_w = nutrient_priorities.get(nutrient, 1.0)
            final_w = base_w * user_w

            micro_scores.append(nutrient_score * final_w)
            micro_weights_list.append(final_w)

    micro_score = float(np.sum(micro_scores) / np.sum(micro_weights_list)) if micro_weights_list else 0.5
    micro_score = float(np.clip(micro_score, 0.0, 1.0))

    # ---------- FINAL COMBINATION ----------
    final_score = float(np.clip(macro_weight * macro_score + micro_weight * micro_score, 0.0, 1.0))

    if verbose:
        debug = {
            "macro_ratio": {k: round(v, 3) for k, v in macro_ratio.items()},
            "macro_score": round(macro_score, 3),
            "micro_score": round(micro_score, 3),
            "final_score": round(final_score, 3),
            "protein_ok": protein_ok,
            "sat_fat_ok": sat_fat_ok,
        }
        return final_score, debug

    return final_score


def score_meal_plan(user_prefs, candidate_meals):
    w = user_prefs["weights"]
    avg_cost_val = avg_cost(candidate_meals)
    cost_score = 0 if avg_cost_val == 0 else normalize(1 / avg_cost_val, inverse=True)

    nutrition_score = evaluate_macros_and_micros(candidate_meals, user_prefs=user_prefs, verbose=False)

    avg_time_val = avg_cook_time(candidate_meals)
    time_score = 0 if avg_time_val == 0 else normalize(1 / avg_time_val, inverse=True)

    diversity_score = diversity_measure(candidate_meals, user_prefs["diversity_rule"])
    leftover_score = leftover_utilization(candidate_meals, user_prefs["prioritize_existing_groceries"])
    preference_score = cuisine_match(user_prefs.get("cuisine_preferences", []), candidate_meals)
    
    total_score = (
        w.get("cost",0) * cost_score
        + w.get("nutrition",0) * nutrition_score
        + w.get("time",0) * time_score
        + w.get("diversity",0) * diversity_score
        + w.get("leftovers",0) * leftover_score
        + w.get("preference",0) * preference_score
    )
    return total_score, nutrition_score


def compute_similarity_matrix(recipes):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"{r['name']}: {r['ingredients']}" for r in recipes]
    embeddings = model.encode(texts, normalize_embeddings=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def compute_candidate_pool_bounds(meal_count, weekly_budget, weekly_variety_factor=2.5):
    effective_meals = int(meal_count * weekly_variety_factor)

    base_min = effective_meals * 5
    base_max = effective_meals * 10

    if weekly_budget < 70:
        base_min *= 1.2
        base_max *= 1.2

    min_pool = int(max(10, base_min))
    max_pool = int(max(min_pool + 5, base_max))

    return min_pool, max_pool


"""
LP optimization on hard contraints of cost and time with optional similarity constraints for diversity.
Shortlists the full recipe catalog down to then be scored and ranked by the more complex FDA score and 
user preference scoring.
"""
def optimize_meal_plan(user_prefs, recipes, use_similarity=True):
    N = len(recipes)
    select = pulp.LpVariable.dicts("select", range(N), 0, 1, cat="Binary")
    y = {}

    model = pulp.LpProblem("MealPlanOptimization", pulp.LpMinimize)
    w = user_prefs["weights"]

    # Only compute similarity matrix if requested
    if use_similarity:
        sim_matrix = compute_similarity_matrix(recipes)
        for i in range(N):
            for j in range(i + 1, N):
                y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", 0, 1, cat="Binary")
                # Linearized diversity constraints
                model += y[(i, j)] <= select[i]
                model += y[(i, j)] <= select[j]
                model += y[(i, j)] >= select[i] + select[j] - 1
    else:
        sim_matrix = np.zeros((N, N))

    # Objective terms
    cost_term = pulp.lpSum(recipes[i].get("cost", 0) * select[i] for i in range(N))
    time_term = pulp.lpSum(recipes[i].get("time", 0) * select[i] for i in range(N))
    if use_similarity:
        diversity_term = pulp.lpSum(sim_matrix[i][j] * y[(i, j)] for i in range(N) for j in range(i + 1, N))
    else:
        diversity_term = 0

    model += w.get("cost",0) * cost_term + w.get("time",0) * time_term + w.get("diversity",0) * diversity_term

    # Meal count & budget constraints
    min_pool, max_pool = compute_candidate_pool_bounds(
        user_prefs["meal_count_per_day"],
        user_prefs["budget_per_week"]
    )
    min_pool = min(min_pool, len(recipes))
    max_pool = min(max_pool, len(recipes))

    model += pulp.lpSum(select[i] for i in range(N)) >= min_pool
    model += pulp.lpSum(select[i] for i in range(N)) <= max_pool
    model += pulp.lpSum(recipes[i].get("cost", 0) * select[i] for i in range(N)) <= (user_prefs["budget_per_week"]/7*user_prefs["meal_count_per_day"])
    
    model.solve()
    lp_shortlist = []
    for i in range(N):
        val = select[i].value()
        if val is not None and val > 0.5:
            lp_shortlist.append(recipes[i])
    return lp_shortlist

""""
Selects the best meal plan from a candidate pool based on nutrition score.
TODO: Change from picking one days worth of meals to picking a full week
"""
def select_best_meal_plan(user_prefs, candidate_pool):
    best_plan = None
    best_score = -1

    k = user_prefs["meal_count_per_day"]

    if len(candidate_pool) < k:
        raise ValueError(
            f"Not enough candidate meals ({len(candidate_pool)}) "
            f"to select {k} meals"
        )

    for plan in itertools.combinations(candidate_pool, k):
        nutrition_score = evaluate_macros_and_micros(
            plan,
            target_macro_ratio=user_prefs["macro_goal_ratio"]
        )
        if nutrition_score > best_score:
            best_score = nutrition_score
            best_plan = plan

    return best_plan


# TODO: Change from csv to json input and also database
def plan_meals(user_prefs, recipe_csv_path, use_similarity=True):
    with open(recipe_csv_path) as f:
        recipes = json.load(f)
    lp_shortlist = optimize_meal_plan(user_prefs, recipes, use_similarity=use_similarity)
    chosen_meals = select_best_meal_plan(user_prefs, lp_shortlist)
    score, fda_score = score_meal_plan(user_prefs, chosen_meals)

    return {
        "chosen_meals": chosen_meals,
        "total_score": score,
        "fda_score": fda_score
    }


# Example usage:
# if __name__ == "__main__":
#     user_prefs = {
#         "meal_count_per_day": 3,
#         "budget_per_week": 50,
#         "weights": {"cost":0.25, "nutrition":0.3, "time":0.2, "diversity":0.1, "leftovers":0.15},
#         "macro_goal_ratio": {"protein":0.3, "fat":0.3, "carbs":0.4},
#         "diversity_rule": True,
#         "prioritize_existing_groceries": True,
#         "cuisine_preferences": ["asian","mexican"],
#     }

#     BASE_DIR = os.path.dirname(__file__)
#     SYNTHETIC_JSON_PATH = os.path.join(BASE_DIR, "tests", "synthetic_recipes.json")

#     # Skip similarity LP for faster testing
#     result = plan_meals(user_prefs, SYNTHETIC_JSON_PATH, use_similarity=False)
#     print(result)
#     print("Chosen meals:", [m["name"] for m in result["chosen_meals"]])
#     print("Meal plan score:", round(result["total_score"],3))
#     print("FDA balance score:", round(result["fda_score"],3))