import streamlit as st
import pandas as pd
import numpy as np
# matplotlib / seaborn imports removed to avoid requiring heavy plotting dependencies
# (add back if you need plots on Streamlit Cloud; include in requirements.txt)
# scikit-learn is optional for advanced predictive features. Import lazily so the app
# can run core pages (search, symptom oracle, batch) even if scikit-learn is not
# installed in the environment. If predictive features are used, we'll check
# `sklearn_available` and show a friendly message when it's unavailable.
sklearn_available = False
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    sklearn_available = True
except Exception:
    sklearn_available = False
import plotly.express as px
import plotly.graph_objects as go
import re
import time
import random

# Set page configuration
st.set_page_config(
    page_title="Ayurvedic Wisdom Portal",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #BB86FC;
        text-align: center;
                    if not handled_by_map:
                        # Instead of running the full corpus symptom-text search here (which is the job
                        # of the Symptom Oracle page), guide the user to the Symptom Oracle where
                        # symptom-detection and multi-symptom prediction are provided in a focused UI.
                        st.info("I couldn't find a disease name that matched your query. For symptom-based detection and multi-symptom matching, please use the Symptom Oracle.")
                        if st.button("Open Symptom Oracle"):
                            st.session_state['goto_symptom_oracle'] = True
                            st.experimental_rerun()
    }
    /* Broader input selectors to handle different Streamlit versions */
    input[type="text"], textarea {
        background-color: rgba(40, 40, 40, 0.95) !important;
        color: #E0E0E0 !important;
        border: 1px solid rgba(187,134,252,0.35) !important;
        padding: 8px !important;
        border-radius: 6px !important;
    }
    input::placeholder, textarea::placeholder {
        color: #BDBDBD !important;
        opacity: 1 !important;
    }
    /* Make buttons more visible */
    .stButton button {
        background-color: rgba(187, 134, 252, 0.2) !important;
        color: #BB86FC !important;
        border: 1px solid #BB86FC !important;
    }
    .stButton button:hover {
        background-color: rgba(187, 134, 252, 0.4) !important;
    }
    /* Opposite card for inverted results */
    .opposite-card {
        background: #111111 !important;
        color: #FFFFFF !important;
        padding: 12px !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
        border: 1px solid #333 !important;
    }
    .opposite-card h3, .opposite-card p, .opposite-card small {
        color: #FFFFFF !important;
    }
    /* Strong rule for prediction/detail cards to ensure white bg + black text */
    .predict-card {
        background: #FFFFFF !important;
        color: #E0E0E0 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
        border: 1px solid #DDD !important;
    }
    .predict-card h3, .predict-card p, .predict-card small {
        color: #E0E0E0 !important;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@500&family=Philosopher&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🧙‍♂️ The Ayurvedic Oracle</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-family: Philosopher, sans-serif; font-size: 1.2em; color: #E0E0E0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>Ancient Wisdom Unlocked Through Modern Knowledge</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/E0E0E0/wizard.png", width=100)
st.sidebar.title("Mystical Navigation")

# Function to load data (supports language selection)
@st.cache_data
def load_data(lang: str = 'en'):
    """Load dataset for the requested language.

    If a Hindi copy does not exist and lang=='hi', create a direct copy of the
    English CSV at `e-ayurveda-solution/AyurGenixAI_Dataset_hi.csv` so the app
    can operate on a separate file. This does not translate content; it only
    creates a per-language copy for downstream processing.
    """
    try:
        import os, shutil
        if lang == 'en':
            path = "e-ayurveda-solution/AyurGenixAI_Dataset.csv"
        else:
            # Hindi variant path
            path = "e-ayurveda-solution/AyurGenixAI_Dataset_hi.csv"
            # If Hindi file doesn't exist, create a copy of the English file so
            # you can edit it independently (for example, to add Hindi translations)
            if not os.path.exists(path):
                src = "e-ayurveda-solution/AyurGenixAI_Dataset.csv"
                if os.path.exists(src):
                    try:
                        shutil.copyfile(src, path)
                    except Exception:
                        # Copy failure may be due to permissions; fall back to src
                        path = src
                else:
                    # English source missing
                    path = src

        df = pd.read_csv(path)
        return df
    except Exception:
        # Silently handle missing/corrupt dataset so the app can still run in
        # symptom→disease-only mode. Do not show a user-facing error here.
        return None

# Let user choose language (English by default). Place this control in the sidebar so it appears early.
use_hindi = st.sidebar.checkbox("Use Hindi corpus (हिन्दी)", value=False, key='use_hindi')
# Allow choosing whether to display Hindi fields in the UI (separate from which corpus to load)
display_hi = st.sidebar.checkbox("Display UI in Hindi (use Disease_hi / Symptoms_hi)", value=False, key='display_hi')
# Paths for mapping files (English and Hindi)
mapping_en_path = r'e-ayurveda-solution/symptom_disease_map_full.csv'
mapping_hi_path = r'e-ayurveda-solution/symptom_disease_map_full_hi.csv'

# Uploader removed per user request — mapping CSVs are auto-detected from the
# `e-ayurveda-solution/` directory instead of being uploaded via the sidebar.
# Load the data for the chosen language
lang = 'hi' if use_hindi else 'en'
df = load_data(lang=lang)

# If dataset failed to load, create a minimal empty DataFrame so the app can
# continue running in symptom→disease only mode. We avoid showing an error
# banner to the user per request.
dataset_missing = False
if df is None:
    import pandas as _pd
    dataset_missing = True
    # minimal useful columns (treatment columns included so Treatment page doesn't crash)
    cols = [
        'Disease', 'Hindi Name', 'Symptoms', 'Disease_hi', 'Symptoms_hi',
        'Ayurvedic Herbs', 'Formulation', 'Diet and Lifestyle Recommendations',
        'Yoga & Physical Therapy', 'Medical Intervention', 'Patient Recommendations'
    ]
    df = _pd.DataFrame(columns=cols)

# Ensure bilingual columns exist so we can switch display easily. If the Hindi columns are missing,
# create them as copies of the English columns (so UI can show something).
if 'Disease_hi' not in df.columns:
    df['Disease_hi'] = df.get('Disease', '')
if 'Symptoms_hi' not in df.columns:
    df['Symptoms_hi'] = df.get('Symptoms', '')

# Create display columns used across the UI. These will be simple views and not used for searching.
if display_hi:
    df['Disease_display'] = df['Disease_hi'].fillna('')
    df['Symptoms_display'] = df['Symptoms_hi'].fillna('')
else:
    df['Disease_display'] = df.get('Disease', '').fillna('')
    df['Symptoms_display'] = df.get('Symptoms', '').fillna('')

# Hindi dataset tools were removed per user request

# Columns to use for searching and token-matching. When display_hi is enabled we
# will search against the display columns (Hindi) so the app behaves in Hindi.
disease_col = 'Disease_display' if display_hi else 'Disease'
symptoms_col = 'Symptoms_display' if display_hi else 'Symptoms'

if dataset_missing:
    # Optionally, suppress pages that require corpus content. For now we keep all
    # pages but avoid showing a fatal error; the Treatment page will simply not
    # find matches and will show no preview unless mapping provides results.
    pass

if df is not None:
    # Add mystical navigation options with icons
    st.sidebar.markdown("<div style='text-align:center; margin-bottom:20px;'>", unsafe_allow_html=True)
    st.sidebar.markdown("<img src='https://img.icons8.com/color/96/E0E0E0/mandala.png' width='80'/>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='font-family:Cinzel, serif; color:#673AB7;'>Pathways to Wisdom</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    sidebar_options = ["🔮 Oracle's Counsel", " Soul Mirror", "☯️ Dosha Harmony", "🌿 Sacred Remedies", "✨ Cosmic Predictions"]
    page = st.sidebar.radio("Choose Your Mystical Path", sidebar_options, key='sidebar_page')
    
    # Map mystical names to actual page names
    page_mapping = {
        "🔮 Oracle's Counsel": "Disease Search",
        " Soul Mirror": "Patient Profiling",
        "☯️ Dosha Harmony": "Dosha Analysis",
        "🌿 Sacred Remedies": "Treatment Recommendations",
        "✨ Cosmic Predictions": "Predictive Analytics"
    }
    
    # Add mystical quotes to sidebar based on selected page
    mystical_quotes = {
        "🔮 Oracle's Counsel": "The Oracle sees through the veil of symptoms to reveal the true nature of ailments.",
        "👤 Soul Mirror": "Know thyself, and the path to wellness shall be revealed.",
        "☯️ Dosha Harmony": "Balance of doshas is the key to harmony within the body, mind, and spirit.",
        "🌿 Sacred Remedies": "Nature provides all remedies; the wise healer knows how to find them.",
        "✨ Cosmic Predictions": "The stars guide our health as they guide our destiny."
    }
    
    st.sidebar.markdown("<div style='background:rgba(103, 58, 183, 0.1); padding:15px; border-radius:10px; margin-top:20px;'>", unsafe_allow_html=True)
    # Use .get to avoid KeyError if the page string contains unexpected characters or encoding
    st.sidebar.markdown(f"<p style='font-family:Philosopher, serif; font-style:italic; text-align:center;'>{mystical_quotes.get(page, '')}</p>", unsafe_allow_html=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # Get the actual page name (use .get to be robust against unexpected sidebar values)
    actual_page = page_mapping.get(page, page)
    # Programmatic navigation flags (kept for backward compatibility)
    # Note: the Symptom Oracle page has been removed from the sidebar.
    if st.session_state.get('goto_symptom_oracle', False):
        # Clear the flag and redirect to Disease Search instead
        st.session_state['goto_symptom_oracle'] = False
        actual_page = "Disease Search"
    # Programmatic redirect to Treatment Recommendations (used when user sends Oracle results)
    if st.session_state.get('goto_treatment', False):
        st.session_state['goto_treatment'] = False
        actual_page = "Treatment Recommendations"
    
    # Disease Search Page
    if actual_page == "Disease Search":
        st.markdown("<h2 class='sub-header'>✨ The Ancient Oracle of Healing</h2>", unsafe_allow_html=True)
        
        import time
        import random
        
        st.markdown("<div class='mystical-card'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: rgba(18, 18, 18, 0.95); border-radius: 16px; padding: 24px; 
        border: 2px solid rgba(187,134,252,0.6); box-shadow: 0 6px 12px rgba(0,0,0,0.45);">
            <p style="color: #F5F5F5; font-size: 1.25em; font-family: 'Philosopher', sans-serif; 
            line-height: 1.7; text-shadow: 0px 1px 3px rgba(0,0,0,0.5); text-align: center; margin:0;">
                <span style="color: #E1BEE7; font-weight: 700; font-size: 1.4em;">Welcome, seeker of ancient wisdom.</span><br>
                <span style="display:block; margin-top:8px; font-size:1.05em; color:#E0E0E0;">I am the Oracle, guardian of five thousand years of Ayurvedic knowledge.</span>
                <span style="display:block; margin-top:6px; font-size:1.05em; color:#E0E0E0;">Share with me the ailment that troubles you, and I shall reveal the secrets of healing passed down through generations.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # Create a text input for disease search with mystical styling optimized for dark background
    if actual_page == "Disease Search":
        st.markdown("<p style='color:#E0E0E0; font-family:Philosopher, serif; margin-bottom:5px;'>Speak the name of the affliction:</p>", unsafe_allow_html=True)
        disease_query = st.text_input("Oracle search", "", key="oracle_input", label_visibility="collapsed", placeholder="e.g. kidney cancer")
        # Ensure 'matches' is always defined to avoid NameError on downstream logic
        try:
            matches = df.iloc[0:0]
        except Exception:
            import pandas as _pd
            matches = _pd.DataFrame()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            search_button = st.button("🔮 Consult the Oracle", key="oracle_button", use_container_width=True)
        # Symptom-mapping mode is compulsory and hidden per user request
        treat_as_symptom = True

        if search_button or disease_query:
            if disease_query:
                # Add a mystical loading effect
                with st.spinner("The Oracle is consulting ancient texts..."):
                    # Artificial delay to create mystical effect
                    time.sleep(1.5)

                # Case-insensitive search (respect display language if selected)
                # Enhanced matching: Unicode-aware normalization + substring + fuzzy matching
                try:
                    import difflib, unicodedata

                    def norm_text(s: str):
                        if s is None:
                            return ""
                        s = str(s)
                        s = unicodedata.normalize('NFKC', s)
                        s = s.strip().lower()
                        s = re.sub(r"\s+", " ", s)
                        return s

                    def compact(s: str):
                        # remove punctuation but preserve Devanagari letters and numbers
                        if s is None:
                            return ""
                        s = norm_text(s)
                        # keep letters, digits and Devanagari block
                        return re.sub(r"[^0-9a-z\u0900-\u097F]", "", s)

                    def find_matches_in_df(query: str):
                        q = norm_text(query)
                        q_compact = compact(query)
                        candidate_cols = []
                        # prefer configured display column
                        candidate_cols.append(disease_col)
                        candidate_cols.extend(['Disease', 'Hindi Name', 'Disease_hi', 'Disease_display'])

                        # Exact normalized match
                        for col in candidate_cols:
                            if col in df.columns:
                                try:
                                    ser = df[col].fillna('').astype(str).apply(norm_text)
                                    mask = ser == q
                                    if mask.any():
                                        return df[mask]
                                except Exception:
                                    continue

                        # Substring match (normalized) across columns
                        masks = []
                        for col in candidate_cols:
                            if col in df.columns:
                                try:
                                    ser = df[col].fillna('').astype(str).apply(norm_text)
                                    mask = ser.str.contains(q, na=False)
                                    masks.append(mask)
                                except Exception:
                                    continue
                        if masks:
                            combined = masks[0]
                            for m in masks[1:]:
                                combined = combined | m
                            if combined.any():
                                return df[combined]

                        # Compact substring (e.g., user typed half-word without spaces)
                        masks = []
                        for col in candidate_cols:
                            if col in df.columns:
                                try:
                                    ser = df[col].fillna('').astype(str).apply(compact)
                                    if q_compact:
                                        mask = ser.str.contains(q_compact, na=False)
                                        masks.append(mask)
                                except Exception:
                                    continue
                        if masks:
                            combined = masks[0]
                            for m in masks[1:]:
                                combined = combined | m
                            if combined.any():
                                return df[combined]

                        # Fuzzy matching: try to find close vocabulary matches
                        vocab = set()
                        for col in candidate_cols:
                            if col in df.columns:
                                try:
                                    vocab.update(df[col].dropna().astype(str).apply(norm_text).unique().tolist())
                                except Exception:
                                    continue
                        vocab_list = list(vocab)
                        if vocab_list:
                            try:
                                candidates = difflib.get_close_matches(q, vocab_list, n=3, cutoff=0.6)
                            except Exception:
                                candidates = []
                            if candidates:
                                # return rows that match the first candidate
                                best = candidates[0]
                                masks = []
                                for col in candidate_cols:
                                    if col in df.columns:
                                        try:
                                            ser = df[col].fillna('').astype(str).apply(norm_text)
                                            mask = ser == best
                                            masks.append(mask)
                                        except Exception:
                                            continue
                                if masks:
                                    combined = masks[0]
                                    for m in masks[1:]:
                                        combined = combined | m
                                    if combined.any():
                                        return df[combined]

                        # no matches
                        return df.iloc[0:0]

                    matches = find_matches_in_df(disease_query)
                except Exception:
                    try:
                        matches = df[df[disease_col].fillna('').str.lower().str.contains(disease_query.lower())]
                    except Exception:
                        matches = df[df['Disease'].str.lower().str.contains(disease_query.lower())]

                # Always consult the precomputed symptom->disease mapping as an "also as symptom" panel.
                mapping_panel = []
                try:
                    import os, csv as _csv, json as _json, glob
                    # choose mapping path depending on selected language
                    # If the expected hi mapping file isn't present, try to auto-detect any Hindi mapping CSV
                    mapping_path = mapping_hi_path if lang == 'hi' else mapping_en_path
                    if lang == 'hi' and not os.path.exists(mapping_path):
                        # look for candidate files in e-ayurveda-solution containing 'hi' or 'symptom' in filename
                        candidates = glob.glob(r"e-ayurveda-solution/*hi*.csv") + glob.glob(r"e-ayurveda-solution/*symptom*.csv")
                        # prefer files with 'symptom' in the name
                        cand = None
                        for c in candidates:
                            base = os.path.basename(c).lower()
                            if 'symptom' in base:
                                cand = c
                                break
                        if not cand and candidates:
                            cand = candidates[0]
                        if cand and os.path.exists(cand):
                            mapping_path = cand
                    syn_path = r'e-ayurveda-solution/synonym_map.json'

                    # Build symptom->disease lookup by scanning for mapping CSVs
                    map_lookup_local = {}
                    clean_map_local = {}
                    mapping_files = []
                    # Prefer the explicit mapping path if present
                    if os.path.exists(mapping_path):
                        mapping_files.append(mapping_path)
                    # Also scan the directory for any CSVs that look like mapping files
                    try:
                        for cand in glob.glob(r"e-ayurveda-solution/*.csv"):
                            if cand not in mapping_files:
                                # quick header sniff to see if it has Symptom and Disease columns
                                try:
                                    with open(cand, 'r', encoding='utf-8') as __fh:
                                        hdr = __fh.readline().lower()
                                        # recognize English headers and common Hindi headers (लक्षण, रोग)
                                        if (('symptom' in hdr or 'symptoms' in hdr or 'लक्षण' in hdr) and 
                                            ('disease' in hdr or 'diseases' in hdr or 'रोग' in hdr)):
                                            mapping_files.append(cand)
                                except Exception:
                                    continue
                    except Exception:
                        pass

                    # Load each discovered mapping file
                    for mf in mapping_files:
                        try:
                            with open(mf, newline='', encoding='utf-8') as _fh:
                                _reader = _csv.DictReader(_fh)
                                fields = [_f.strip() for _f in (_reader.fieldnames or [])]
                                # detect which field names to use for symptom and disease
                                symptom_field = None
                                disease_field = None
                                for f in fields:
                                    fl = f.lower()
                                    if fl in ('symptom', 'symptoms') or 'लक्षण' in f:
                                        symptom_field = f
                                    if fl in ('disease', 'diseases') or 'रोग' in f:
                                        disease_field = f
                                # fallback: if there are exactly two columns, treat first as symptom, second as disease
                                if not symptom_field or not disease_field:
                                    if len(fields) >= 2:
                                        symptom_field = symptom_field or fields[0]
                                        disease_field = disease_field or fields[1]

                                for _r in _reader:
                                    _sym = str(_r.get(symptom_field,'')).strip() if symptom_field else ''
                                    _dis = str(_r.get(disease_field,'')).strip() if disease_field else ''
                                    if not _sym:
                                        continue
                                    k = re.sub(r"\s+", " ", _sym.lower())
                                    map_lookup_local.setdefault(k, []).append(_dis)
                                    # use Unicode-aware cleaning so non-Latin scripts (e.g., Devanagari) are preserved
                                    ck = re.sub(r"[^\w]", "", k)
                                    if ck:
                                        clean_map_local.setdefault(ck, []).append(k)
                        except Exception:
                            # ignore unreadable files
                            continue

                    q_norm = re.sub(r"\s+", " ", disease_query.strip().lower())
                    # Unicode-aware cleaning so Hindi characters are preserved for matching
                    q_clean = re.sub(r"[^\w]", "", q_norm)
                    # If no direct found_map, try fuzzy matching on mapping keys as well
                    found_map = []
                    if map_lookup_local and q_norm in map_lookup_local:
                        found_map = map_lookup_local.get(q_norm, [])
                    if not found_map and map_lookup_local:
                        partial_keys = [k for k in map_lookup_local.keys() if q_norm and (q_norm in k or (q_clean and q_clean in re.sub(r"[^\w]","", k)))]
                        if not partial_keys and q_clean and q_clean in clean_map_local:
                            partial_keys = list(clean_map_local.get(q_clean, []))
                        if partial_keys:
                            seen = []
                            for k in partial_keys:
                                for d in map_lookup_local.get(k, []):
                                    if d not in seen:
                                        seen.append(d)
                            found_map = seen
                    # If still no mapping found, try fuzzy matching on mapping keys (helps with partial/half-typed Hindi)
                    if not found_map and map_lookup_local:
                        try:
                            import difflib, unicodedata

                            def _norm(s):
                                if s is None:
                                    return ""
                                s = str(s)
                                s = unicodedata.normalize('NFKC', s)
                                s = s.strip().lower()
                                s = re.sub(r"\s+", " ", s)
                                return s

                            keys = list(map_lookup_local.keys())
                            keys_norm_map = { _norm(k): k for k in keys }
                            qn = _norm(q_norm)
                            cand_norms = difflib.get_close_matches(qn, list(keys_norm_map.keys()), n=6, cutoff=0.6)
                            if cand_norms:
                                seen = []
                                for kn in cand_norms:
                                    orig_k = keys_norm_map.get(kn)
                                    if not orig_k:
                                        continue
                                    for d in map_lookup_local.get(orig_k, []):
                                        if d not in seen:
                                            seen.append(d)
                                if seen:
                                    found_map = seen
                        except Exception:
                            pass

                    # disease->symptom synonym translation (help when user typed a disease but we want symptom tokens)
                    if not found_map and os.path.exists(syn_path):
                        try:
                            with open(syn_path, 'r', encoding='utf-8') as _sf:
                                _syn = _json.load(_sf)
                                mapped = _syn.get(disease_query.strip().lower())
                                if mapped:
                                    first = mapped[0]
                                    nk = re.sub(r"\s+", " ", first.strip().lower())
                                    if nk in map_lookup_local:
                                        found_map = map_lookup_local[nk]
                        except Exception:
                            pass

                    if found_map:
                        # Filter mapping results so we only return diseases that exist in the main dataset.
                        def _filter_to_corpus(disease_list):
                            out = []
                            # candidate columns to check in the corpus
                            candidate_cols = ['Disease', 'Hindi Name', 'Disease_hi', 'Disease_display']
                            for d in disease_list:
                                if not d:
                                    continue
                                d_norm = str(d).strip().lower()
                                matched = False
                                # exact match first
                                for col in candidate_cols:
                                    if col in df.columns:
                                        try:
                                            series = df[col].fillna('').astype(str).str.lower().str.strip()
                                            mask = series == d_norm
                                            if mask.any():
                                                r = df[mask].iloc[0]
                                                canonical = r.get('Disease_display', r.get('Disease', d))
                                                if canonical not in out:
                                                    out.append(canonical)
                                                matched = True
                                                break
                                        except Exception:
                                            continue
                                if matched:
                                    continue
                                # partial contains fallback
                                for col in candidate_cols:
                                    if col in df.columns:
                                        try:
                                            series = df[col].fillna('').astype(str).str.lower()
                                            mask = series.str.contains(d_norm, na=False)
                                            if mask.any():
                                                r = df[mask].iloc[0]
                                                canonical = r.get('Disease_display', r.get('Disease', d))
                                                if canonical not in out:
                                                    out.append(canonical)
                                                matched = True
                                                break
                                        except Exception:
                                            continue
                                # if still not matched we skip this mapping result (user requested corpus-only answers)
                            return out

                        mapping_panel = _filter_to_corpus(found_map)
                except Exception:
                    mapping_panel = []

            # Debug diagnostics have been removed from the UI per user request.
            # Mapping happens silently using mapping CSVs discovered in `e-ayurveda-solution/`.

            skip_render_matches = False
            # Keep track of diseases already rendered on this search to avoid duplicates
            displayed_diseases = set()
            if treat_as_symptom:
                # If user explicitly asked to treat input as symptom, render mapping results prominently and skip disease-name rendering
                if mapping_panel:
                    try:
                        # (no automatic transfer) — show mapping results and provide an explicit send button below
                        st.markdown("<div style='text-align:center; padding:10px;'><img src='https://img.icons8.com/fluency/96/E0E0E0/crystal-ball.png' width='60'/></div>", unsafe_allow_html=True)
                        st.markdown(f"<h3 class='wisdom-text'>Symptom search: I found {len(mapping_panel)} disease(s) linked to '{disease_query}'. Showing examples below:</h3>", unsafe_allow_html=True)
                        for d in mapping_panel[:40]:
                            dn = str(d).strip().lower()
                            if dn in displayed_diseases:
                                continue
                            displayed_diseases.add(dn)
                            st.markdown(f"<div style='padding-left:12px; color:#E0E0E0; background:rgba(255,255,255,0.04); margin:4px 0; border-radius:4px; padding:8px;'>{d}</div>", unsafe_allow_html=True)
                        # After listing mapping examples, offer an explicit send button so the user can transfer them
                        try:
                            send_key = f"send_map_{re.sub(r'\W+', '_', disease_query.strip().lower())}"
                            if st.button("➡️ Send these suggested diseases to Sacred Remedies", key=send_key):
                                st.session_state['oracle_selected_diseases'] = mapping_panel[:40]
                                try:
                                    st.session_state['treatment_focus'] = mapping_panel[0]
                                except Exception:
                                    pass
                                # programmatic navigation: set sidebar selection then navigate
                                st.session_state['sidebar_page'] = "🌿 Sacred Remedies"
                                st.session_state['goto_treatment'] = True
                                st.experimental_rerun()
                        except Exception:
                            pass
                        skip_render_matches = True
                    except Exception:
                        skip_render_matches = False
                else:
                    # No symptom->disease mapping found — attempt automatic fallback:
                    # if the input appears to be a disease name, show disease results instead
                    try:
                        reverse_panel = []
                        q_low = disease_query.strip().lower()
                        # Try exact matches against multiple possible disease name columns
                        candidate_cols = []
                        # prefer the configured disease_col first
                        candidate_cols.append(disease_col)
                        # also check canonical English and Hindi name columns if present
                        candidate_cols.extend(['Disease', 'Hindi Name', 'Disease_hi', 'Disease_display'])

                        exact = None
                        for col in candidate_cols:
                            try:
                                if col in df.columns:
                                    series = df[col].fillna('').astype(str).str.lower().str.strip()
                                    exact_rows = df[series == q_low]
                                    if not exact_rows.empty:
                                        exact = exact_rows
                                        break
                            except Exception:
                                continue

                        if exact is not None and not exact.empty:
                            for _, r in exact.iterrows():
                                dd = r.get('Disease_display', r.get('Disease', 'Unknown'))
                                if dd and dd not in reverse_panel:
                                    reverse_panel.append(dd)
                        else:
                            # try partial contains across candidate columns
                            partial_rows = None
                            for col in candidate_cols:
                                try:
                                    if col in df.columns:
                                        series = df[col].fillna('').astype(str).str.lower()
                                        pr = df[series.str.contains(q_low, na=False)]
                                        if partial_rows is None:
                                            partial_rows = pr
                                        else:
                                            # union of rows
                                            partial_rows = pd.concat([partial_rows, pr]).drop_duplicates()
                                except Exception:
                                    continue

                            if partial_rows is not None and not partial_rows.empty:
                                for _, r in partial_rows.iterrows():
                                    dd = r.get('Disease_display', r.get('Disease', 'Unknown'))
                                    if dd and dd not in reverse_panel:
                                        reverse_panel.append(dd)

                        if reverse_panel:
                            mapping_panel = reverse_panel
                            st.info("Input looks like a disease name — showing disease results as a fallback.")
                        else:
                            st.info("No symptom→disease mapping found for this input; falling back to disease-name search.")
                    except Exception:
                        st.info("No symptom→disease mapping found for this input; falling back to disease-name search.")

            if not skip_render_matches and not matches.empty:
                # Mystical success message
                st.markdown("<div style='text-align:center; padding:10px;'><img src='https://img.icons8.com/fluency/96/E0E0E0/crystal-ball.png' width='60'/></div>", unsafe_allow_html=True)

                # Display each matching disease in a mystical way
                # track the diseases we render so we can expose them to other pages
                rendered_diseases = []
                # also track canonical disease names to avoid duplicate full-card rendering
                rendered_canonical = set()
                for idx, row in matches.iterrows():
                    st.markdown("<div class='oracle-response'>", unsafe_allow_html=True)

                    # Prepare display strings depending on the bilingual switch
                    disease_disp = row.get('Disease_display', row.get('Disease', 'Unknown'))
                    # canonical disease name used for deduplication
                    canonical_name = str(row.get('Disease', disease_disp)).strip().lower()
                    if canonical_name in rendered_canonical:
                        # skip rendering duplicate disease cards
                        continue
                    rendered_canonical.add(canonical_name)
                    symptoms_disp = row.get('Symptoms_display', row.get('Symptoms', ''))

                    # Add mystical introduction (use display label)
                    mystical_intros = [
                        f"The ancient scrolls speak of {disease_disp}...",
                        f"I have consulted the stars about {disease_disp}...",
                        f"The wisdom of a thousand healers reveals the nature of {disease_disp}...",
                        f"From the depths of ancient knowledge, I bring forth the truth about {disease_disp}..."
                    ]

                    st.markdown(f"""<h3 style='text-align:center; font-family:Cinzel, serif; color:#512DA8; 
                        padding: 15px; background-color: rgba(103, 58, 183, 0.1); 
                        border-radius: 10px; margin-bottom: 20px; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);'>
                        {random.choice(mystical_intros)}
                        </h3>""", unsafe_allow_html=True)

                    # Enhanced visibility for the oracle's response - breaking into smaller sections for better rendering

                    # Basic information
                    st.markdown("""<div class='wisdom-text'>""", unsafe_allow_html=True)

                    # Section 1: Basic information
                    st.markdown(f"""
                    <div style="background-color: rgba(103, 58, 183, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                        In the ancient language of Sanskrit, this affliction was known by many names, including {row.get('Hindi Name')} in Hindi and {row.get('Marathi Name')} in Marathi traditions.

                        Those afflicted with this condition often experience {symptoms_disp}. The ancient healers would confirm this through {row.get('Diagnosis & Tests')}.
                    </div>
                    """, unsafe_allow_html=True)
                    # Treatment recommendation column removed per user request.  
                    # (Previously: explanation + button linking to Treatment Recommendations page.)

                    # Close wisdom-text div
                    st.markdown("""</div>""", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                    # If we also found mapping-based results (user input looks like a symptom), show them below as an "Also interpreted as a symptom" panel
                    try:
                        if mapping_panel:
                            total_map = len(mapping_panel)
                            examples_map = mapping_panel[:28]
                            st.markdown("<hr style='border-color:rgba(255,255,255,0.06)'/>", unsafe_allow_html=True)
                            # (message removed by user request)
                            for d in examples_map:
                                dn = str(d).strip().lower()
                                if dn in displayed_diseases:
                                    continue
                                displayed_diseases.add(dn)
                                rendered_diseases.append(d)
                                st.markdown(f"<div style='padding-left:12px; color:#E0E0E0; background:rgba(255,255,255,0.03); margin:4px 0; border-radius:4px; padding:8px;'>{d}</div>", unsafe_allow_html=True)
                            # Provide an explicit button to send these suggested diseases to Sacred Remedies
                            try:
                                send_key = f"send_map_{re.sub(r'\W+', '_', disease_query.strip().lower())}"
                                if st.button("➡️ Send these suggested diseases to Sacred Remedies", key=send_key):
                                    st.session_state['oracle_selected_diseases'] = examples_map
                                    # optionally preselect the first disease in Treatment Recommendations
                                    try:
                                        st.session_state['treatment_focus'] = examples_map[0]
                                    except Exception:
                                        pass
                                    st.session_state['sidebar_page'] = "🌿 Sacred Remedies"
                                    st.session_state['goto_treatment'] = True
                                    st.experimental_rerun()
                            except Exception:
                                pass
                    except Exception:
                        pass
                else:
                    # If disease-name search had no matches, but our precomputed mapping found symptom->disease entries,
                    # surface them here as a helpful "Also interpreted as a symptom" panel and skip the token corpus search.
                    # Ensure tokens is always defined so later checks don't throw NameError
                    tokens = []
                    handled_by_map = False
                    if mapping_panel:
                        try:
                            total = len(mapping_panel)
                            examples = mapping_panel[:28]
                            st.markdown("<div style='text-align:center; padding:10px;'><img src='https://img.icons8.com/fluency/96/E0E0E0/crystal-ball.png' width='60'/></div>", unsafe_allow_html=True)
                            # (message removed by user request)
                            for d in examples:
                                dn = str(d).strip().lower()
                                if dn in displayed_diseases:
                                    continue
                                displayed_diseases.add(dn)
                                st.markdown(f"<div style='padding-left:12px; color:#E0E0E0; background:rgba(255,255,255,0.04); margin:4px 0; border-radius:4px; padding:8px;'>{d}</div>", unsafe_allow_html=True)
                            handled_by_map = True
                        except Exception:
                            handled_by_map = False

                    if not handled_by_map:
                        # fallback to corpus symptom-text search (existing behaviour)
                        tokens = [p.strip() for p in re.split(',|;|/|\\band\\b', disease_query) if p.strip()]
                    if len(tokens) > 0:
                        sel_lower = [t.lower() for t in tokens]
                        results = []
                        for _, rrow in df.iterrows():
                            # Respect the selected symptoms column (Hindi display if chosen)
                            sym_text = str(rrow.get(symptoms_col, '')).lower()
                            if not sym_text:
                                continue
                            matched = [s for s in sel_lower if s in sym_text]
                            if matched:
                                score = len(matched) / len(sel_lower)
                                results.append({'score': score, 'matched': matched, 'row': rrow})

                        if results:
                            # aggregate unique disease names preserving order
                            disease_names = []
                            for res in sorted(results, key=lambda x: x['score'], reverse=True):
                                name = res['row'].get('Disease', 'Unknown')
                                if name not in disease_names:
                                    disease_names.append(name)

                            total = len(disease_names)
                            examples = disease_names[:28]

                            st.markdown("<div style='text-align:center; padding:10px;'><img src='https://img.icons8.com/fluency/96/E0E0E0/crystal-ball.png' width='60'/></div>", unsafe_allow_html=True)
                            st.markdown(f"<p class='wisdom-text'>I couldn't find a disease name that exactly matched '{disease_query}' — so I searched the symptom corpus and found {total} disease(s) mentioning these symptom token(s). Showing examples below:</p>", unsafe_allow_html=True)

                            # Show count and list of example diseases (plain list)
                            st.markdown(f"<p style='font-family:Philosopher, serif; color:#E0E0E0;'>{tokens[0]} matches: {total} disease(s) (examples)</p>", unsafe_allow_html=True)
                            for d in examples:
                                # Use subtle dark example chips instead of stark white bars
                                st.markdown(f"<div style='padding-left:12px; color:#E0E0E0; background:rgba(255,255,255,0.04); margin:4px 0; border-radius:4px; padding:8px;'>{d}</div>", unsafe_allow_html=True)

                            # Offer an explicit send button so the user can transfer these example diseases
                            try:
                                send_key2 = f"send_examples_{re.sub(r'\W+', '_', tokens[0].strip().lower())}"
                                if st.button("➡️ Send these example diseases to Sacred Remedies", key=send_key2):
                                    st.session_state['oracle_selected_diseases'] = examples
                                    try:
                                        st.session_state['treatment_focus'] = examples[0]
                                    except Exception:
                                        pass
                                    st.session_state['sidebar_page'] = "🌿 Sacred Remedies"
                                    st.session_state['goto_treatment'] = True
                                    st.experimental_rerun()
                            except Exception:
                                pass

                            # Also show the top matched rows with match percentage (compact)
                            for res in sorted(results, key=lambda x: x['score'], reverse=True)[:10]:
                                r = res['row']
                                pct = int(res['score'] * 100)
                                # compute colors for this card (no per-expander invert here)
                                # Safely resolve color helpers/variables that may not exist in other pages
                                invert_flag = locals().get('invert_results', False)
                                compute_fn = locals().get('_compute_colors', None)
                                # determine bg choice (use Diabetes highlight if applicable)
                                disease_name = str(r.get('Disease','')).lower()
                                diabetes_flag = locals().get('diabetes_highlight', False)
                                diabetes_color = locals().get('diabetes_bg_color', None)
                                if diabetes_flag and diabetes_color and 'diabetes' in disease_name:
                                    bg_choice = diabetes_color
                                else:
                                    bg_choice = locals().get('card_bg_color', '#FFFFFF')
                                if callable(compute_fn):
                                    try:
                                        bg, text, border = compute_fn(bg_choice, invert=invert_flag)
                                    except Exception:
                                        bg, text, border = ('#FFFFFF', '#E0E0E0', '#DDD')
                                else:
                                    if invert_flag:
                                        bg, text, border = ('#111111', '#FFFFFF', '#333')
                                    else:
                                        bg, text, border = (bg_choice, '#E0E0E0', '#DDD')
                                # Use CSS class when global invert is requested to force inverted colors
                                if locals().get('invert_results', False):
                                    st.markdown("<div class='opposite-card'>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div style='background:{bg}; color:{text}; padding:12px; border-radius:8px; margin-bottom:12px; border:1px solid {border};'>", unsafe_allow_html=True)
                                # Use the display column when available
                                disp_name = r.get('Disease_display', r.get('Disease', 'Unknown'))
                                st.markdown(f'<h3 style="font-family:Cinzel, serif; color:{text}; margin:0 0 6px 0">{disp_name} <small style="color:{"#333333" if text=="#E0E0E0" else "#BBBBBB"}; font-size:0.85rem;">({pct}% symptom match)</small></h3>', unsafe_allow_html=True)
                                st.markdown(f"<p style='color:{text}; margin:4px 0;'>Matched tokens: {', '.join(res['matched'])}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='color:{text}; margin:4px 0;'>Key Symptoms: {r.get('Symptoms','N/A')}</p>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align:center; padding:20px; background-color:rgba(30, 30, 30, 0.7); border-radius:15px; border:1px solid rgba(187, 134, 252, 0.3);'>", unsafe_allow_html=True)
                            # question-mark icon removed per user request
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='text-align:center; padding:20px; background-color:rgba(30, 30, 30, 0.7); border-radius:15px; border:1px solid rgba(187, 134, 252, 0.3);'>", unsafe_allow_html=True)
                        # question-mark icon removed per user request
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; padding:20px; background-color:rgba(30, 30, 30, 0.5); border-radius:15px; font-family:Philosopher, serif; font-size:1.1em; font-style:italic; color:#BB86FC;'>Speak the name of the affliction, and I shall reveal the ancient wisdom...</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Symptom Oracle page removed per user request. It is no longer available from the sidebar.
    # Previously this block implemented the Symptom Oracle (predict-by-symptoms).
    # If you need a compact helper to run predictions programmatically, use `_run_prediction_for` defined elsewhere.

    # Ancient Knowledge (formerly Data Overview)
    if actual_page == "Data Overview":
        st.markdown("<h2 class='sub-header'>🔮 The Ancient Scrolls of Knowledge</h2>", unsafe_allow_html=True)
        
        # Introduction
        st.markdown("<div class='mystical-card'>", unsafe_allow_html=True)
        st.markdown("<p class='wisdom-text'>Welcome to the sacred library of Ayurvedic knowledge, where the wisdom of thousands of years has been preserved. Here you can glimpse the breadth of ancient healing traditions without revealing their deepest secrets.</p>", unsafe_allow_html=True)
        
        # Interactive elements
        knowledge_type = st.selectbox(
            "Which ancient wisdom interests you?", 
            ["The Breadth of Knowledge", "Elemental Categories", "Wisdom of the Healers"]
        )
        
        if knowledge_type == "The Breadth of Knowledge":
            # Knowledge summary
            st.markdown("<div class='oracle-response'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:Cinzel, serif;'>The Scope of Ancient Wisdom</h3>", unsafe_allow_html=True)
            
            # Mystical representation of data stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div style='text-align:center; padding:30px;'>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-family:Cinzel, serif; color:#673AB7;'>{df.shape[0]}</h2>", unsafe_allow_html=True)
                st.markdown("<p>Ancient healing traditions documented across centuries</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='text-align:center; padding:30px;'>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='font-family:Cinzel, serif; color:#673AB7;'>{df.shape[1]}</h2>", unsafe_allow_html=True)
                st.markdown("<p>Aspects of knowledge preserved in each tradition</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Visual representation instead of data
            import plotly.express as px
            categories = ['Vata', 'Pitta', 'Kapha', 'Combination']
            values = [df[df['Doshas'].str.contains('Vata')].shape[0], 
                      df[df['Doshas'].str.contains('Pitta')].shape[0], 
                      df[df['Doshas'].str.contains('Kapha')].shape[0], 
                      df[~(df['Doshas'].str.contains('Vata') | df['Doshas'].str.contains('Pitta') | df['Doshas'].str.contains('Kapha'))].shape[0]]
            
            fig = px.pie(
                names=categories, 
                values=values, 
                title="Distribution of Dosha Knowledge",
                color_discrete_sequence=px.colors.sequential.Agsunset
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        elif knowledge_type == "Elemental Categories":
            # Categories of knowledge
            st.markdown("<div class='oracle-response'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:Cinzel, serif;'>Categories of Ancient Wisdom</h3>", unsafe_allow_html=True)
            
            # Show categories instead of data types
            cols = ["Ayurvedic Herbs", "Formulation", "Doshas", "Diet and Lifestyle Recommendations", "Yoga & Physical Therapy"]
            
            for col in cols:
                with st.expander(f"📜 {col}"):
                    unique_vals = df[col].unique()
                    samples = unique_vals[:min(5, len(unique_vals))]
                    
                    st.markdown("<div class='wisdom-text'>", unsafe_allow_html=True)
                    st.markdown(f"<p>Examples of {col.lower()} found in ancient texts:</p>", unsafe_allow_html=True)
                    for sample in samples:
                        st.markdown(f"• *{sample}*")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:  # Wisdom of the Healers
            # Interactive showcase of sample knowledge
            st.markdown("<div class='oracle-response'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:Cinzel, serif;'>Glimpses of Healing Traditions</h3>", unsafe_allow_html=True)
            
            # Random selection instead of showing the dataframe
            import random
            sample_size = min(5, len(df))
            samples = df.sample(sample_size)
            
            for idx, row in samples.iterrows():
                disease_disp = row.get('Disease_display', row.get('Disease', 'Unknown'))
                symptoms_disp = row.get('Symptoms_display', row.get('Symptoms', ''))
                with st.expander(f"Ancient wisdom about {disease_disp}"):
                    st.markdown(f"""<div class='wisdom-text'>
                        <p>The ancient healers recognized {disease_disp} by these signs: <em>{symptoms_disp}</em></p>
                        
                        <p>They would prescribe {row.get('Ayurvedic Herbs')} in this sacred formulation: <em>{row.get('Formulation')}</em></p>
                        
                        <p>Their guidance for recovery: <em>{row.get('Diet and Lifestyle Recommendations')}</em></p>
                    </div>""", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Secret download option hidden behind mystical interaction
        with st.expander("🔐 For Initiates Only"):
            st.markdown("<p style='font-family:Philosopher, serif;'>The ancient texts can be transcribed for those deemed worthy...</p>", unsafe_allow_html=True)
            password = st.text_input("Speak the sacred password", type="password")
            if password == "ayurveda":
                st.download_button(
                    label="🧙‍♂️ Receive the Ancient Scrolls",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='AncientAyurvedicTexts.csv',
                    mime='text/csv',
                )
            elif password:
                st.markdown("<p style='color:#ff5555;'>The spirits do not recognize this incantation...</p>", unsafe_allow_html=True)
    
    # Wisdom Exploration Page (formerly Exploratory Analysis)
    elif actual_page == "Exploratory Analysis":
        st.markdown("<h2 class='sub-header'>🧠 Exploring the Ancient Patterns</h2>", unsafe_allow_html=True)
        
        # Introduction with mystical language
        st.markdown("<div class='mystical-card'>", unsafe_allow_html=True)
        st.markdown("<p class='wisdom-text'>The ancient healers discovered patterns in nature that reveal the hidden connections between remedies, symptoms, and the cosmic elements. Explore these mystic patterns to uncover deeper knowledge of healing traditions.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Select features for analysis (hidden as mystical patterns)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Visualization options with mystical naming
        viz_type = st.selectbox("Choose Your Path of Enlightenment", ["Elemental Distributions", "Cosmic Connections", "The Classification of Being"])
        
        if viz_type == "Elemental Distributions":
            st.markdown("<div class='oracle-response'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:Cinzel, serif;'>The Balance of Elements</h3>", unsafe_allow_html=True)
            st.markdown("<p class='wisdom-text'>Ancient healers understood that all life exists in a balance of energies. See how these energies distribute across different aspects of healing.</p>", unsafe_allow_html=True)
            
            if numeric_cols:
                mystical_names = {
                    "Symptom Severity": "Intensity of Affliction",
                    "Duration of Treatment": "Time of Healing Journey",
                    "Age Group": "Life's Progression",
                    # Add more mappings as needed
                }
                
                # Create display names for selection
                display_names = [mystical_names.get(col, col) for col in numeric_cols]
                name_to_col = dict(zip(display_names, numeric_cols))
                
                selected_display = st.selectbox("Select an energy pattern to explore", display_names)
                selected_col = name_to_col[selected_display]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram with mystical theme
                    fig = px.histogram(df, x=selected_col, marginal="box", 
                                      title=f"Flow of {selected_display}",
                                      color_discrete_sequence=["#673AB7"],
                                      template="plotly_dark")
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.03)",
                        font=dict(family="Philosopher, serif")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box Plot with mystical theme
                    fig = px.box(df, y=selected_col, 
                                title=f"Cosmic Distribution of {selected_display}",
                                color_discrete_sequence=["#9C27B0"],
                                template="plotly_dark")
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.03)",
                        font=dict(family="Philosopher, serif")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary reframed as mystical insight
                st.markdown("<h4 style='font-family:Cinzel, serif;'>Ancient Measurements</h4>", unsafe_allow_html=True)
                
                stats = df[selected_col].describe()
                
                mystical_stats = {
                    "count": "Number of documented cases",
                    "mean": "Balance point of energy",
                    "std": "Variance in cosmic flow",
                    "min": "Lowest observed energy",
                    "25%": "First quarter of the cycle",
                    "50%": "Median point of balance",
                    "75%": "Third quarter of the cycle",
                    "max": "Highest observed energy"
                }
                
                stats_df = pd.DataFrame({
                    "Mystical Measure": list(mystical_stats.values()),
                    "Value": stats.values
                })
                
                st.table(stats_df)
            else:
                st.markdown("<p style='font-style:italic;'>The ancient texts contain no numerical measurements for this phenomenon.</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif viz_type == "Cosmic Connections":
            st.markdown("<div class='oracle-response'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:Cinzel, serif;'>The Cosmic Web of Interconnection</h3>", unsafe_allow_html=True)
            st.markdown("<p class='wisdom-text'>In ancient Ayurveda, all things are connected through invisible threads. This sacred map reveals how different aspects of healing are intertwined in the cosmic dance.</p>", unsafe_allow_html=True)
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                # Mystical Heatmap
                fig = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    color_continuous_scale='Magma',
                    title="Sacred Map of Interconnections",
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.03)",
                    font=dict(family="Philosopher, serif", color="#E0E0E0")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature correlations with mystical names
                mystical_names = {
                    "Symptom Severity": "Intensity of Affliction",
                    "Duration of Treatment": "Time of Healing Journey",
                    "Age Group": "Life's Progression",
                    # Add more mappings as needed
                }
                
                # Create display names for selection
                display_names = [mystical_names.get(col, col) for col in numeric_cols]
                name_to_col = dict(zip(display_names, numeric_cols))
                
                selected_display = st.selectbox("Select an aspect to see its cosmic connections", display_names)
                selected_feature = name_to_col[selected_display]
                
                corr_series = corr_matrix[selected_feature].sort_values(ascending=False).drop(selected_feature)
                
                # Create mystical display names for correlated features
                corr_display = {col: mystical_names.get(col, col) for col in corr_series.index}
                corr_display_names = [corr_display[col] for col in corr_series.index]
                
                # Bar chart with mystical theme
                fig = px.bar(
                    x=corr_series.values,
                    y=corr_display_names,
                    orientation='h',
                    title=f"Cosmic Forces Connected with {selected_display}",
                    labels={'x': 'Strength of Connection', 'y': 'Aspects of Being'},
                    color=corr_series.values,
                    color_continuous_scale="Magma",
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.03)",
                    font=dict(family="Philosopher, serif", color="#E0E0E0")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Mystical interpretation
                strongest_corr = corr_series.abs().nlargest(1)
                if not strongest_corr.empty:
                    strongest_feat = strongest_corr.index[0]
                    strongest_val = strongest_corr.values[0]
                    corr_type = "harmony" if strongest_val > 0 else "opposition"
                    
                    st.markdown(f"""<div class='wisdom-text' style='background: rgba(103, 58, 183, 0.1); padding: 15px; border-radius: 10px; border-left: 3px solid #9C27B0;'>
                        <p>The ancient texts reveal that {selected_display} exists in {corr_type} with {corr_display[strongest_feat]}.</p>
                        <p style='font-style:italic;'>When one increases, the other {'also rises' if strongest_val > 0 else 'diminishes'}. This cosmic balance has been used by healers for millennia to restore harmony.</p>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-style:italic;'>The cosmic threads require at least two measured aspects to reveal their connections.</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif viz_type == "The Classification of Being":
            st.markdown("<div class='oracle-response'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-family:Cinzel, serif;'>The Ancient Classifications</h3>", unsafe_allow_html=True)
            st.markdown("<p class='wisdom-text'>The sages categorized all elements of existence. Explore how they divided the healing arts into sacred categories that guide practitioners to this day.</p>", unsafe_allow_html=True)
            
            if categorical_cols:
                # Create mystical category names
                mystical_categories = {
                    "Disease": "Afflictions of Being",
                    "Doshas": "Primal Forces",
                    "Ayurvedic Herbs": "Sacred Plants",
                    "Formulation": "Alchemical Recipes",
                    "Constitution/Prakriti": "Soul Templates",
                    "Diet and Lifestyle Recommendations": "Path to Balance",
                    "Yoga & Physical Therapy": "Body Harmonization",
                    # Add more mappings as needed
                }
                
                # Create display names for selection
                display_names = [mystical_categories.get(col, col) for col in categorical_cols]
                cat_to_col = dict(zip(display_names, categorical_cols))
                
                selected_display = st.selectbox("Select a sacred classification to explore", display_names)
                selected_cat = cat_to_col[selected_display]
                
                # Bar chart of category counts with mystical theme
                value_counts = df[selected_cat].value_counts().reset_index()
                value_counts.columns = [selected_cat, 'Frequency']
                
                # Limit to top 10 categories if there are many
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                    st.markdown("<p style='font-style:italic; font-size:0.9em;'>Showing the 10 most common sacred classifications...</p>", unsafe_allow_html=True)
                
                fig = px.bar(
                    value_counts, 
                    x=selected_cat, 
                    y='Frequency',
                    title=f"Distribution of {selected_display}",
                    color=selected_cat,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.03)",
                    font=dict(family="Philosopher, serif", color="#E0E0E0")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Interactive exploration
                st.markdown("<h4 style='font-family:Cinzel, serif;'>Sacred Patterns</h4>", unsafe_allow_html=True)
                st.markdown("<p>Choose a category to reveal its hidden wisdom:</p>", unsafe_allow_html=True)
                
                categories = value_counts[selected_cat].tolist()
                if len(categories) > 0:
                    selected_value = st.selectbox("Select a specific classification", categories)
                    
                    # Show examples of this category
                    examples = df[df[selected_cat] == selected_value].sample(min(3, df[df[selected_cat] == selected_value].shape[0]))
                    
                    st.markdown(f"""<div style='background: rgba(103, 58, 183, 0.1); padding: 15px; border-radius: 10px; border-left: 3px solid #9C27B0;'>
                        <h4 style='font-family:Cinzel, serif;'>The Wisdom of {selected_value}</h4>
                    </div>""", unsafe_allow_html=True)
                    
                    for idx, row in examples.iterrows():
                        disp = row.get('Disease_display', row.get('Disease', 'Unknown'))
                        herbs = row.get('Ayurvedic Herbs', '')
                        dietrec = row.get('Diet and Lifestyle Recommendations', '')
                        st.markdown(f"""<div class='wisdom-text' style='margin: 10px 0; padding: 10px; border-bottom: 1px solid rgba(103, 58, 183, 0.3);'>
                            <p>{disp} is treated with <em>{herbs}</em>.</p>
                            <p>The ancient recommendation: <em>"{dietrec}"</em></p>
                        </div>""", unsafe_allow_html=True)
                
                # If there are numeric columns, allow mystical analysis by category
                if numeric_cols:
                    selected_num = st.selectbox("Select a numeric column to analyze by category", numeric_cols)
                    
                    fig = px.box(
                        df, 
                        x=selected_cat, 
                        y=selected_num,
                        title=f"{selected_num} by {selected_cat}",
                        color=selected_cat
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No categorical columns found in the dataset.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Patient Profiling Page
    elif actual_page == "Patient Profiling":
        st.markdown("<h2 class='sub-header'>Patient Profiling</h2>", unsafe_allow_html=True)
        
        # Patient profile form
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Enter Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with col2:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
            
        # Additional symptoms/conditions (this would depend on your actual dataset columns)
        st.write("### Symptoms and Conditions")
        
        # This is a placeholder - replace with actual columns from your dataset
        symptom_options = ["Fever", "Cough", "Headache", "Fatigue", "Joint Pain", "Digestive Issues"]
        selected_symptoms = st.multiselect("Select Symptoms", symptom_options)
        
        stress_level = st.slider("Stress Level", 0, 10, 5)
        sleep_quality = st.slider("Sleep Quality", 0, 10, 7)
        
        # Create patient profile
        if st.button("Generate Patient Profile"):
            bmi = weight / ((height/100) ** 2)
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### Patient Profile")
            
            profile_col1, profile_col2 = st.columns(2)
            
            with profile_col1:
                st.write(f"**Name:** {name}")
                st.write(f"**Age:** {age}")
                st.write(f"**Gender:** {gender}")
                st.write(f"**BMI:** {bmi:.2f}")
                
                # BMI Category
                bmi_category = ""
                if bmi < 18.5:
                    bmi_category = "Underweight"
                elif bmi < 25:
                    bmi_category = "Normal"
                elif bmi < 30:
                    bmi_category = "Overweight"
                else:
                    bmi_category = "Obese"
                
                st.write(f"**BMI Category:** {bmi_category}")
            
            with profile_col2:
                st.write(f"**Weight:** {weight} kg")
                st.write(f"**Height:** {height} cm")
                st.write(f"**Stress Level:** {stress_level}/10")
                st.write(f"**Sleep Quality:** {sleep_quality}/10")
                
                if selected_symptoms:
                    st.write(f"**Symptoms:** {', '.join(selected_symptoms)}")
                else:
                    st.write("**Symptoms:** None reported")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Dosha Analysis Page
    elif actual_page == "Dosha Analysis":
        st.markdown("<h2 class='sub-header'>Dosha Analysis</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("""
        ### Understanding Doshas in Ayurveda
        
        In Ayurvedic medicine, doshas are the three energies that define a person's makeup:
        
        - **Vata** (Air & Space): Controls bodily functions associated with motion
        - **Pitta** (Fire & Water): Controls metabolic systems
        - **Kapha** (Earth & Water): Controls growth and maintains structure
        
        Complete the questionnaire below to determine your dominant dosha.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Dosha questionnaire
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Dosha Questionnaire")
        
        st.write("#### Physical Characteristics")
        body_frame = st.radio("Body Frame", ["Thin, lean (Vata)", "Medium, athletic (Pitta)", "Large, solid (Kapha)"])
        weight = st.radio("Weight", ["Difficult to gain (Vata)", "Stable, moderate (Pitta)", "Gains easily, difficult to lose (Kapha)"])
        skin = st.radio("Skin Type", ["Dry, rough (Vata)", "Warm, reddish, sensitive (Pitta)", "Thick, oily, cool (Kapha)"])
        
        st.write("#### Mental Characteristics")
        mind = st.radio("Mind", ["Quick, adaptable, restless (Vata)", "Sharp, focused, determined (Pitta)", "Calm, steady, slow to anger (Kapha)"])
        stress = st.radio("Under Stress", ["Anxious, worried (Vata)", "Irritable, aggressive (Pitta)", "Withdrawn, complacent (Kapha)"])
        
        st.write("#### Behavioral Patterns")
        sleep = st.radio("Sleep Pattern", ["Light, interrupted (Vata)", "Moderate, sharp waking (Pitta)", "Deep, difficult to wake (Kapha)"])
        appetite = st.radio("Appetite", ["Variable, irregular (Vata)", "Strong, sharp (Pitta)", "Steady, can skip meals (Kapha)"])
        
        # Calculate dosha scores
        if st.button("Analyze My Dosha"):
            responses = [body_frame, weight, skin, mind, stress, sleep, appetite]
            
            vata_count = sum(1 for r in responses if "Vata" in r)
            pitta_count = sum(1 for r in responses if "Pitta" in r)
            kapha_count = sum(1 for r in responses if "Kapha" in r)
            
            total = vata_count + pitta_count + kapha_count
            vata_pct = (vata_count / total) * 100
            pitta_pct = (pitta_count / total) * 100
            kapha_pct = (kapha_count / total) * 100
            
            # Determine dominant dosha
            scores = {"Vata": vata_pct, "Pitta": pitta_pct, "Kapha": kapha_pct}
            dominant_dosha = max(scores, key=scores.get)
            
            # Display results
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### Your Dosha Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Dosha Distribution")
                fig = px.pie(
                    values=[vata_pct, pitta_pct, kapha_pct],
                    names=["Vata", "Pitta", "Kapha"],
                    title="Your Dosha Composition"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("#### Your Dominant Dosha")
                st.write(f"### {dominant_dosha}")
                
                if dominant_dosha == "Vata":
                    st.write("""
                    **Vata dominant individuals** are typically creative, energetic, and quick-thinking.
                    
                    **Health Focus:**
                    - Maintain regular routines
                    - Stay warm and grounded
                    - Favor warm, cooked foods
                    - Practice calming activities like meditation
                    """)
                elif dominant_dosha == "Pitta":
                    st.write("""
                    **Pitta dominant individuals** are typically intelligent, focused, and ambitious.
                    
                    **Health Focus:**
                    - Avoid excessive heat
                    - Include cooling foods in diet
                    - Manage stress through moderation
                    - Make time for relaxation
                    """)
                else:  # Kapha
                    st.write("""
                    **Kapha dominant individuals** are typically strong, stable, and compassionate.
                    
                    **Health Focus:**
                    - Stay active with regular exercise
                    - Seek variety and stimulation
                    - Favor light, warm, and dry foods
                    - Create change and excitement in routines
                    """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Treatment Recommendations
    elif actual_page == "Treatment Recommendations":
        st.markdown("<h2 class='sub-header'>Ayurvedic Treatment Recommendations</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Offer two modes: by common concern (classic flow) or by selecting a specific disease from the dataset
        mode = st.radio("Lookup mode", ["By Concern", "By Disease"], index=0, horizontal=True)

        # Common health concerns (kept for convenience)
        health_concerns = [
            "Digestive Issues", 
            "Stress & Anxiety", 
            "Joint Pain", 
            "Skin Problems", 
            "Respiratory Issues",
            "Sleep Disorders",
            "Weight Management",
            "Fatigue & Low Energy"
        ]

        if mode == "By Concern":
            st.write("### Health Concern Selection")
            selected_concern = st.selectbox("Select Your Health Concern", health_concerns)
            st.write("### Personalize Your Recommendations")
            dosha_type = st.selectbox("Select Your Dominant Dosha", ["Vata", "Pitta", "Kapha", "I don't know"])
            severity = st.slider("Severity of Condition", 1, 10, 5)
            duration = st.selectbox("Duration of Condition", ["Recent (< 1 month)", "Short-term (1-6 months)", "Chronic (> 6 months)"])
            if st.button("Generate Recommendations"):
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write(f"### Recommendations for: {selected_concern}")

            # Dietary recommendations (show when mode is By Concern)
            st.write("#### Dietary Recommendations")

            if selected_concern == "Digestive Issues":
                if dosha_type == "Vata":
                    st.write("""
                    - Favor warm, cooked, and easy-to-digest foods
                    - Include warming spices like ginger, cumin, and fennel
                    - Avoid cold, raw foods and irregular eating patterns
                    - Try a tea of cumin, coriander, and fennel seeds
                    """)
                elif dosha_type == "Pitta":
                    st.write("""
                    - Favor cooling, sweet, and bitter foods
                    - Include mint, coriander, and fennel in your diet
                    - Avoid spicy, sour, and fermented foods
                    - Try aloe vera juice before meals
                    """)
                elif dosha_type == "Kapha":
                    st.write("""
                    - Favor light, warm, and spicy foods
                    - Include plenty of bitter greens and pungent spices
                    - Avoid heavy, oily, and sweet foods
                    - Try ginger tea with honey and lemon
                    """)
                else:
                    st.write("""
                    - Establish regular eating times
                    - Eat in a calm environment without distractions
                    - Include a variety of spices like ginger, cumin, and fennel
                    - Drink warm water throughout the day
                    """)

            # Herbal remedies section
            st.write("#### Herbal Remedies")

            if selected_concern == "Digestive Issues":
                st.write("""
                - **Triphala**: A balanced formula for digestive health
                - **Hingvastak**: Helps improve digestion and reduce gas
                - **Avipattikar**: Balances stomach acid and supports healthy digestion
                """)
            elif selected_concern == "Stress & Anxiety":
                st.write("""
                - **Ashwagandha**: Adaptogen that helps the body manage stress
                - **Brahmi**: Supports mental clarity and calm
                - **Jatamansi**: Promotes relaxation and healthy sleep
                """)
            elif selected_concern == "Joint Pain":
                st.write("""
                - **Turmeric**: Anti-inflammatory properties support joint health
                - **Boswellia**: Helps maintain healthy joints and mobility
                - **Shallaki**: Supports healthy joint function
                """)
            else:
                st.write("""
                - **Triphala**: Supports overall health and balance
                - **Ashwagandha**: Adaptogenic herb that supports vitality
                - **Turmeric**: Supports healthy inflammatory response
                """)

            # Lifestyle recommendations
            st.write("#### Lifestyle Recommendations")

            if dosha_type == "Vata":
                st.write("""
                - Maintain regular daily routines
                - Practice gentle yoga and meditation
                - Keep warm and avoid excessive cold
                - Use warm oil massage (Abhyanga) regularly
                """)
            elif dosha_type == "Pitta":
                st.write("""
                - Avoid excessive heat and sun exposure
                - Practice cooling yoga and meditation
                - Make time for relaxation and leisure
                - Enjoy walks in nature during cooler times of day
                """)
            elif dosha_type == "Kapha":
                st.write("""
                - Maintain an active lifestyle with regular exercise
                - Practice stimulating yoga and pranayama
                - Wake up early and avoid daytime naps
                - Use dry brushing to stimulate circulation
                """)
            else:
                st.write("""
                - Establish a consistent daily routine
                - Include regular physical activity
                - Practice stress management techniques
                - Ensure adequate rest and sleep
                """)

            # Disclaimer
            st.info("**Disclaimer**: These recommendations are for informational purposes only and should not replace professional medical advice. Please consult with a qualified Ayurvedic practitioner before starting any new health regimen.")

        else:
            # By Disease mode: allow picking a disease from the dataset. If the disease was selected previously
            # from the Disease Search, pre-fill from session_state['treatment_focus'] if present.
            st.write("### Select a disease to view its treatment recommendations (from the corpus)")
            pre = st.session_state.get('treatment_focus', None)

            # If the Oracle previously showed a list of diseases, use that as the suggestion list.
            oracle_suggestions = st.session_state.get('oracle_selected_diseases', None)

            # Build the selectable disease list: put the oracle-suggested items first (if present),
            # otherwise fall back to a small static suggestion set for dizziness.
            if oracle_suggestions and isinstance(oracle_suggestions, (list, tuple)) and len(oracle_suggestions) > 0:
                suggested = list(oracle_suggestions)
            else:
                suggested = [
                    "Arrhythmia",
                    "Heart Attack (Myocardial Infarction)",
                    "Hypertension",
                    "Hypoglycemia",
                    "Nipah Virus Infection",
                    "Pulmonary Hypertension",
                    "Atherosclerosis",
                    "Adrenal Insufficiency",
                    "Angina",
                    "Aortic Aneurysm",
                    "Polycythemia Vera",
                ]

            all_diseases = sorted(df['Disease'].dropna().unique().tolist())
            # If the Oracle has suggested diseases, show only those suggestions here.
            # Otherwise, include the static suggestions followed by the full dataset list.
            ordered = []
            if oracle_suggestions and isinstance(oracle_suggestions, (list, tuple)) and len(oracle_suggestions) > 0:
                for s in suggested:
                    if s and s not in ordered:
                        ordered.append(s)
            else:
                # include static suggestions first, then all diseases from corpus
                for s in suggested:
                    if s and s not in ordered:
                        ordered.append(s)
                for d in all_diseases:
                    if d not in ordered:
                        ordered.append(d)

            if pre and pre in ordered:
                sel_index = ordered.index(pre)
                selected_disease = st.selectbox("Choose Disease", ordered, index=sel_index)
            else:
                selected_disease = st.selectbox("Choose Disease", ordered)

            if st.button("Show treatment for selected disease"):
                import difflib

                # helper to normalize strings
                def norm(s):
                    if s is None:
                        return ""
                    return str(s).lower().strip()

                query = norm(selected_disease)

                # determine candidate name columns: English 'Disease' plus any column with 'hindi' or Devanagari marker
                name_cols = ['Disease']
                for c in df.columns:
                    cl = c.lower()
                    if 'hindi' in cl or 'हिंदी' in c:
                        if c not in name_cols:
                            name_cols.append(c)

                # Attempt 1: exact match across name columns
                mask = None
                for c in name_cols:
                    try:
                        col_series = df[c].astype(str).fillna('').str.lower().str.strip()
                    except Exception:
                        continue
                    m = col_series == query
                    if mask is None:
                        mask = m
                    else:
                        mask = mask | m

                rows = df[mask] if mask is not None else df.iloc[0:0]

                # Attempt 2: contains
                if rows.empty:
                    mask = None
                    for c in name_cols:
                        try:
                            col_series = df[c].astype(str).fillna('').str.lower()
                        except Exception:
                            continue
                        m = col_series.str.contains(query)
                        if mask is None:
                            mask = m
                        else:
                            mask = mask | m
                    rows = df[mask] if mask is not None else df.iloc[0:0]

                # Attempt 3: fuzzy match on name vocabulary
                best_match = None
                if rows.empty:
                    vocab = []
                    for c in name_cols:
                        try:
                            vocab.extend(df[c].dropna().astype(str).unique().tolist())
                        except Exception:
                            continue
                    # use get_close_matches (case-sensitive input but we provide original vocab)
                    try:
                        candidates = difflib.get_close_matches(selected_disease, vocab, n=3, cutoff=0.6)
                    except Exception:
                        candidates = []
                    if candidates:
                        best_match = candidates[0]
                        # find rows matching that best_match across name_cols
                        mask = None
                        for c in name_cols:
                            try:
                                col_series = df[c].astype(str).fillna('').str.lower().str.strip()
                            except Exception:
                                continue
                            m = col_series == norm(best_match)
                            if mask is None:
                                mask = m
                            else:
                                mask = mask | m
                        rows = df[mask] if mask is not None else df.iloc[0:0]

                if not rows.empty:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    display_title = selected_disease
                    if best_match and best_match.lower().strip() != selected_disease.lower().strip():
                        display_title = f"{selected_disease} (matched to: {best_match})"
                    st.write(f"### Treatment recommendations for: {display_title}")

                    # Show a compact preview of the matched row so user can see which dataset row was used
                    preview_cols = ['Disease'] + [c for c in name_cols if c != 'Disease']
                    treatment_cols = [
                        'Ayurvedic Herbs', 'Formulation', 'Diet and Lifestyle Recommendations',
                        'Yoga & Physical Therapy', 'Medical Intervention', 'Patient Recommendations'
                    ]
                    show_cols = [c for c in preview_cols + treatment_cols if c in df.columns]
                    try:
                        st.write("**Matched dataset row (preview):**")
                        st.dataframe(rows[show_cols].iloc[0:1].fillna(''))
                    except Exception:
                        # fallback to simple text preview
                        r0 = rows.iloc[0]
                        for c in show_cols:
                            st.write(f"- {c}: {r0.get(c, '')}")

                    # Show combined info from the first row
                    r = rows.iloc[0]
                    herbs = r.get('Ayurvedic Herbs', '')
                    form = r.get('Formulation', '')
                    diet = r.get('Diet and Lifestyle Recommendations', '')
                    yoga = r.get('Yoga & Physical Therapy', '')
                    med = r.get('Medical Intervention', '')
                    patient_rec = r.get('Patient Recommendations', '')

                    if herbs:
                        st.write("#### Ayurvedic Herbs:")
                        st.write(herbs)
                    if form:
                        st.write("#### Formulation / Preparation:")
                        st.write(form)
                    if diet:
                        st.write("#### Diet & Lifestyle Recommendations:")
                        st.write(diet)
                    if yoga:
                        st.write("#### Yoga & Physical Therapy:")
                        st.write(yoga)
                    if med:
                        st.write("#### When to seek modern medical care:")
                        st.write(med)
                    if patient_rec:
                        st.write("#### Patient Recommendations:")
                        st.write(patient_rec)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No detailed treatment information found in the dataset for this disease. Try a slightly different name or check the dataset preview to see if a matching row exists.")
            
            
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Predictive Analytics page removed per user request

else:
    # Dataset failed to load earlier — suppress the visible error per user request.
    # The app will continue running in symptom→disease-only mode using any
    # symptom->disease mappings present in the repository.
    pass

# Footer
# Static inverted footer (no controls)
st.markdown("""
<div style='text-align: center; margin-top: 30px; padding: 20px; background-color: #111111; color: #FFFFFF; border-radius: 10px;'>
    <p style='margin:0; font-weight:600;'>AyurGenixAI Dashboard - Combining Ancient Wisdom with Modern Technology</p>
    <p style='font-size: 0.8rem; margin:4px 0 0 0;'>© 2025 AyurGenixAI</p>
</div>
""", unsafe_allow_html=True)