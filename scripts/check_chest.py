import pandas as pd
import re

path = r'E:/cloning wala projects/ayurved part 2/e-ayurveda-solution/AyurGenixAI_Dataset.csv'

df = pd.read_csv(path)
raw_symptoms = df['Symptoms'].dropna().astype(str).tolist()
symptom_tokens = []
for s in raw_symptoms:
    parts = [p.strip() for p in re.split(',|;|/|\\band\\b', s) if p.strip()]
    symptom_tokens.extend(parts)

unique_symptoms = sorted({t for t in symptom_tokens if len(t) > 1})

matches = [s for s in unique_symptoms if 'chest' in s.lower()]
print('Found', len(matches), 'unique symptom tokens containing "chest":')
for m in matches:
    print(repr(m))

# Also show diseases whose Symptoms column contains 'chest pain'
cp = 'chest pain'
matched_rows = df[df['Symptoms'].fillna('').str.lower().str.contains(re.escape(cp))]
print('\nRows with exact "chest pain" in Symptoms column:', len(matched_rows))
print(matched_rows[['Disease','Symptoms']].head(10).to_string(index=False))
