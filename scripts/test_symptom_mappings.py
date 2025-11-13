import re
import pandas as pd

CSV_PATH = "e-ayurveda-solution/AyurGenixAI_Dataset.csv"

def load_df(path=CSV_PATH):
    df = pd.read_csv(path)
    return df


def build_unique_symptoms(df):
    raw_symptoms = df['Symptoms'].dropna().astype(str).tolist()
    symptom_tokens = []
    for s in raw_symptoms:
        parts = [p.strip() for p in re.split(',|;|/|\\band\\b', s) if p.strip()]
        symptom_tokens.extend(parts)
    unique_symptoms = sorted({t for t in symptom_tokens if len(t) > 1})
    return unique_symptoms


def predict(df, symptom_list, top_k=5, match_mode='Any'):
    sel_lower = [s.lower() for s in symptom_list if s]
    results = []
    for idx, row in df.iterrows():
        sym_text = str(row.get('Symptoms','')).lower()
        if not sym_text:
            continue
        matched_tokens = [s for s in sel_lower if s in sym_text]
        match_count = len(matched_tokens)
        if match_count > 0:
            if match_mode == 'All' and match_count < len(sel_lower):
                score = (match_count / len(sel_lower)) * 0.7
            else:
                score = match_count / len(sel_lower)
            results.append({'disease': row.get('Disease','N/A'), 'score': score, 'matched': matched_tokens})
    results.sort(key=lambda x: x['score'], reverse=True)
    # aggregate unique disease preserving order
    seen = set()
    agg = []
    for r in results:
        if r['disease'] not in seen:
            seen.add(r['disease'])
            agg.append(r)
        if len(agg) >= top_k:
            break
    return agg


if __name__ == '__main__':
    df = load_df()
    print(f"Loaded dataset with {len(df)} rows")

    # common symptoms to check
    common = ["fatigue", "fever", "abdominal pain", "headache", "cough", "nausea", "dizziness", "chest pain"]

    for s in common:
        preds = predict(df, [s], top_k=5)
        print('\n' + '='*60)
        print(f"Symptom: {s} -> {len(preds)} prediction(s)")
        for p in preds:
            print(f"  - {p['disease']} ({int(p['score']*100)}%) matched: {', '.join(p['matched'])}")

    # sample of unique symptoms
    unique = build_unique_symptoms(df)
    print(f"\nUnique symptom tokens found: {len(unique)}. Showing sample checks for first 12 tokens...\n")
    for token in unique[:12]:
        preds = predict(df, [token], top_k=3)
        if preds:
            print(f"{token} -> {len(preds)} predictions; top: {preds[0]['disease']} ({int(preds[0]['score']*100)}%)")
        else:
            print(f"{token} -> no predictions")
