import csv, re

MAP_PATH = r'e-ayurveda-solution/symptom_disease_map_full.csv'

map_lookup = {}
clean_map = {}
all_keys = []

with open(MAP_PATH, newline='', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        s = str(r.get('Symptom','')).strip()
        d = str(r.get('Disease','')).strip()
        if not s:
            continue
        k = re.sub(r"\s+"," ", s.lower())
        map_lookup.setdefault(k, []).append(d)
        all_keys.append(k)
        ck = re.sub(r"[^a-z0-9]","", k)
        if ck:
            clean_map.setdefault(ck, []).append(k)

queries = [
    'fatigue',
    'chest pain',
    'chest-pain',
    'chest',
    'headache',
    'head ache',
    'migraine',
    'abdominal pain',
]

print(f"Loaded {len(all_keys)} mapping keys, {sum(len(v) for v in map_lookup.values())} entries")

for q in queries:
    q_norm = re.sub(r"\s+"," ", q.strip().lower())
    q_clean = re.sub(r"[^a-z0-9]","", q_norm)
    found = None
    branch = None
    if q_norm in map_lookup:
        found = map_lookup[q_norm]
        branch = 'exact'
    else:
        # partial by normalized substring
        partial = [k for k in map_lookup.keys() if q_norm and q_norm in k]
        # cleaned-match
        if not partial and q_clean and q_clean in clean_map:
            partial = list(clean_map.get(q_clean, []))
            branch = 'cleaned'
        elif partial:
            branch = 'partial'
        if partial:
            # aggregate unique diseases from partial keys
            seen = []
            for k in partial:
                for d in map_lookup.get(k, []):
                    if d not in seen:
                        seen.append(d)
            found = seen

    print('\nQuery:', q)
    print(' q_norm:', q_norm, ' q_clean:', q_clean)
    print(' branch:', branch)
    if found:
        print(' found count:', len(found), ' example:', found[:6])
    else:
        # try showing first few keys that contain q_norm or q_clean
        similar = [k for k in all_keys if q_norm in k or (q_clean and q_clean in re.sub(r'[^a-z0-9]','', k))]
        print(' found: None; similar keys sample:', similar[:8])

# show some keys that include 'chest' and 'head' to inspect variants
print('\nSample keys containing "chest" ->', [k for k in all_keys if 'chest' in k][:20])
print('Sample keys containing "head" ->', [k for k in all_keys if 'head' in k][:20])
