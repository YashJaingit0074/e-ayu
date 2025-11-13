"""Generate a simple synonym map from disease -> common symptom tokens.

This script reads `e-ayurveda-solution/AyurGenixAI_Dataset.csv` and creates
`e-ayurveda-solution/synonym_map.json` mapping disease lowercased names to
the most frequent symptom tokens that appear in the Symptoms column for that
disease. The output is intentionally conservative (top 1-3 tokens) and can be
used by the app to translate disease-name inputs like "migraine" -> "headache".

Run: python scripts/generate_synonym_map.py
"""
import json
import re
from collections import Counter, defaultdict
import pandas as pd

DATA_CSV = r'e-ayurveda-solution/AyurGenixAI_Dataset.csv'
OUT_JSON = r'e-ayurveda-solution/synonym_map.json'

def normalize_token(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).strip().lower())

def tokenize_symptoms(s: str):
    parts = [p.strip() for p in re.split(',|;|/|\\band\\b', str(s)) if p and p.strip()]
    return [normalize_token(p) for p in parts if p and p.strip()]

def main():
    df = pd.read_csv(DATA_CSV)
    disease_to_tokens = defaultdict(Counter)

    for _, r in df.iterrows():
        disease = str(r.get('Disease','')).strip()
        symptoms = str(r.get('Symptoms','')).strip()
        if not disease or not symptoms:
            continue
        disease_l = disease.lower()
        toks = tokenize_symptoms(symptoms)
        for t in toks:
            disease_to_tokens[disease_l][t] += 1

    synonym_map = {}
    for disease_l, counter in disease_to_tokens.items():
        if not counter:
            continue
        most = counter.most_common(5)
        selected = []
        for tok, cnt in most:
            if cnt >= 2 or len(selected) == 0:
                selected.append(tok)
        if selected:
            synonym_map[disease_l] = selected[:3]

    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(synonym_map, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(synonym_map)} disease->symptom mappings to {OUT_JSON}")

if __name__ == '__main__':
    main()
