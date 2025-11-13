import re
import pandas as pd

CSV_PATH = "e-ayurveda-solution/AyurGenixAI_Dataset.csv"
OUT_PATH = "e-ayurveda-solution/symptom_disease_map_full.csv"


def load_df(path=CSV_PATH):
    df = pd.read_csv(path)
    return df


def build_unique_symptoms(df):
    raw_symptoms = df['Symptoms'].dropna().astype(str).tolist()
    symptom_tokens = []
    for s in raw_symptoms:
        parts = [p.strip() for p in re.split(',|;|/|\\band\\b', s) if p.strip()]
        symptom_tokens.extend(parts)
    unique_symptoms = sorted({t for t in symptom_tokens if len(t) > 0})
    return unique_symptoms


def build_mapping(df, unique_symptoms):
    rows = []
    unmapped = []
    for token in unique_symptoms:
        token_lower = token.lower()
        try:
            matches = df[df['Symptoms'].fillna('').str.lower().str.contains(re.escape(token_lower))]
        except Exception:
            matches = pd.DataFrame()
        if not matches.empty:
            diseases = matches['Disease'].dropna().unique().tolist()
            for d in diseases:
                rows.append({'Symptom': token, 'Disease': d})
        else:
            unmapped.append(token)
    map_df = pd.DataFrame(rows)
    return map_df, unmapped


if __name__ == '__main__':
    df = load_df()
    unique = build_unique_symptoms(df)
    print(f"Loaded df with {len(df)} rows; unique symptom tokens: {len(unique)}")
    map_df, unmapped = build_mapping(df, unique)
    print(f"Mapped tokens: {len(unique) - len(unmapped)}; Unmapped tokens: {len(unmapped)}")
    if len(unmapped) > 0:
        print("Sample unmapped tokens:", unmapped[:30])
    try:
        map_df.to_csv(OUT_PATH, index=False)
        print(f"Wrote mapping rows: {len(map_df)} to {OUT_PATH}")
    except Exception as e:
        print(f"Could not write output CSV: {e}")
