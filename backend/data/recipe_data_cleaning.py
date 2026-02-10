import re
import ast
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import os

INPUT_CSV = "recipes_data.csv"
OUT_DIR = "parquet_clean"
os.makedirs(OUT_DIR, exist_ok=True)

writer = None
chunk_size = 50_000
id_counter = 1

# TODO: #8 might want to expand stopwords in the future
STOPWORDS = {
    "fresh", "large", "small", "medium", "chopped", "diced", "minced",
    "to", "taste", "and", "or", "optional", "temperature", "handful",
    "sliced", "ground", "grated", "shredded", "crushed", "whole", "only"
}

UNIT_WORDS = {
    "cup", "cups", "tbsp", "tablespoon", "tablespoons",
    "tsp", "teaspoon", "teaspoons",
    "oz", "ounce", "ounces", "lb", "pound", "pounds",
    "grams", "g", "kg", "ml", "l", "liter", "liters",
    "pinch", "clove", "cloves", "can", "cans", 
    "handful", "bunch", "bunches", "piece", "pieces"
}

def normalize_ingredient(text: str):
    if not text or not isinstance(text, str):
        return None

    text = text.lower()
    text = re.sub(r"\([^)]*\)", "", text)   
    text = re.sub(r"[^a-z\s]", "", text)   

    tokens = [
        t for t in text.split()
        if t not in STOPWORDS and t not in UNIT_WORDS
    ]

    if not tokens:
        return None

    return "_".join(tokens[:3])  # keep core ingredient phrase

def parse_ingredients(raw):
    if raw is None:
        return []

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        raw = raw.strip()

        # Try Python list
        if raw.startswith("["):
            try:
                items = ast.literal_eval(raw)
            except Exception:
                items = raw.split(",")
        else:
            items = raw.split(",")
    else:
        return []

    normed = []
    for item in items:
        ing = normalize_ingredient(item)
        if ing:
            normed.append(ing)

    return list(set(normed))  # dedupe

for i, chunk in enumerate(
    tqdm(pd.read_csv(INPUT_CSV, chunksize=chunk_size))
):
    
    keep_cols = []
    for c in ["title", "ingredients", "directions", "NER"]:
        if c in chunk.columns:
            keep_cols.append(c)

    chunk = chunk[keep_cols].copy()
    chunk = chunk.rename(columns={"NER": "ingredients_raw"})
    chunk = chunk.rename(columns={"title": "name"})

    if "ingredients_raw" in chunk.columns:
        chunk = chunk[
            chunk["ingredients_raw"].notna() &
            chunk["ingredients_raw"].astype(str).str.strip().ne("")
        ]

    chunk["ingredients_norm"] = chunk["ingredients_raw"].apply(parse_ingredients)
    chunk["ingredient_count"] = chunk["ingredients_norm"].apply(len)

    chunk = chunk[
        (chunk["ingredient_count"] >= 2) &
        (chunk["ingredient_count"] <= 25)
    ]

    chunk = chunk.reset_index(drop=True)
    chunk["id"] = range(id_counter, id_counter + len(chunk))
    id_counter += len(chunk)

    table = pa.Table.from_pandas(chunk, preserve_index=False)

    out_path = f"{OUT_DIR}/recipes_cleaned.parquet"
    if writer is None:
        writer = pq.ParquetWriter(
            out_path,
            table.schema,
            compression="zstd"
        )

    writer.write_table(table)

if writer:
    writer.close()

