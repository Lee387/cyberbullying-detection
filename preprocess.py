import pandas as pd
import re
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ─── 1. LOAD & MERGE ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Loading and merging datasets")
print("=" * 60)

files = [
    "/mnt/user-data/uploads/data_twitter_Cyberbullying_Gabungan_5Tahun.xls",
    "/mnt/user-data/uploads/data_twitter_Cyberbullying_Part2.xls",
    "/mnt/user-data/uploads/data_twitter_Cyberbullying_Part3.xls",
]

dfs = []
for f in files:
    try:
        df = pd.read_csv(f, sep='\t', encoding='utf-8', on_bad_lines='skip')
    except:
        df = pd.read_csv(f, sep='\t', encoding='latin-1', on_bad_lines='skip')
    print(f"  Loaded {f.split('/')[-1]}: {len(df)} rows")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
print(f"\n  Total merged rows: {len(merged)}")
print(f"  Columns: {list(merged.columns)}")

# Drop rows with missing tweet text
merged = merged.dropna(subset=['Text_Tweet']).reset_index(drop=True)
print(f"  After dropping NaN tweets: {len(merged)} rows")

# ─── MAP KEYWORDS TO 6 CYBERBULLYING CATEGORIES ─────────────────────────────────
print("\n" + "=" * 60)
print("MAPPING: Keywords → 6 Cyberbullying Categories")
print("=" * 60)

category_map = {
    # 1. Harsh Words & Animal Insults
    "Umpatan Kasar & Binatang": [
        "tolol", "goblok", "bego", "idiot", "anjing", "babi", "bangsat", "bajingan", "kampang",
        "ngentot", "kontol", "memek", "dongo", "dungu", "kampret", "kampretos", "monyet",
        "kunyuk", "asu", "tai", "berak", "jancuk", "keparat", "bacot", "peju", "silit", "pantat"
    ],
    # 2. Body Shaming & Physical Conditions
    "Body Shaming & Kondisi Fisik": [
        "gendut", "jelek banget", "muka burik", "dekil", "kurus kering", "item dekil",
        "muka plastik", "kayak babi", "bogel", "kuntet", "gembrot", "ceking", "tepos", "tonggos",
        "muka aspal", "burik banget", "jerawatan", "keliatan tua", "cacat", "bisu", "budek", "kurap"
    ],
    # 3. Moral Degradation & Sexual Harassment
    "Degradasi Moral & Pelecehan Seksual": [
        "jablay", "murahan", "lonte", "banci", "cegil", "anak haram", "sampah masyarakat",
        "cewek gatel", "simpenan", "pelakor", "jual diri", "murahan banget", "ani-ani", "lont",
        "gatel banget", "kegatelan", "bencong", "lesbi", "maho", "pecun", "pelacur", "perek", "bandot"
    ],
    # 4. Psychological Attacks & Mockery
    "Serangan Psikologis & Sindiran": [
        "mati aja", "bunuh diri aja", "gak berguna", "nyusahin", "mending mati", "hapus akun",
        "kena mental", "dihujat netizen", "dibully", "pansos", "sok asik", "sok iye", "sok suci",
        "sok pintar", "pick me", "caper", "caper banget", "sok asik lu", "cancel aja", "najis",
        "jijik", "jijik banget", "gak tahu malu", "muka tembok", "norak", "picik", "plinplan", "sombong"
    ],
    # 5. Intellectual & Social Degradation
    "Merendahkan Intelektual & Sosial": [
        "jamet", "sdm rendah", "otak udang", "gak punya otak", "tolol banget", "bego lu",
        "otak kopong", "bocil kematian", "miskin", "gembel", "kismin", "modal utang", "kampungan",
        "udik", "anak yatim", "kere", "kuli", "ampas", "bocah", "copet", "sarap", "sinting", "edan",
        "geblek", "gila", "autis"
    ],
    # 6. Political / SARA / Labeling
    "Politik / SARA / Labeling": [
        "bong", "cebong", "cebonger", "cebongers", "onta", "kadrun", "kafir", "kapir", "komunis",
        "pki", "radikal", "rezim", "antek", "asing", "sontoloyo", "tapir", "tuyul", "kuntilanak", "iblis", "setan"
    ]
}

# Build reverse lookup: keyword → category
keyword_to_cat = {}
for cat, keywords in category_map.items():
    for kw in keywords:
        keyword_to_cat[kw.lower().strip()] = cat

# Map each row
def map_category(keyword):
    kw = str(keyword).lower().strip()
    return keyword_to_cat.get(kw, "Tidak Terklasifikasi")

merged['Kategori_Cyberbullying'] = merged['Kategori_Keyword'].apply(map_category)

cat_dist = merged['Kategori_Cyberbullying'].value_counts()
print("\n  Category distribution:")
for cat, count in cat_dist.items():
    print(f"    {cat}: {count}")

# ─── 2. TEXT PREPROCESSING ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Text Cleaning")
print("=" * 60)

def clean_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)        # Remove URLs
    text = re.sub(r'@\w+', '', text)                          # Remove mentions
    text = re.sub(r'#\w+', '', text)                          # Remove hashtags
    text = re.sub(r'<[^>]+>', '', text)                       # Remove HTML tags
    text = re.sub(r'RT\s*:', '', text)                        # Remove RT markers
    text = re.sub(r'[^\w\s]', '', text)                       # Remove punctuation/special chars
    text = re.sub(r'\d+', '', text)                           # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()                  # Normalize whitespace
    return text

merged['Text_Cleaned'] = merged['Text_Tweet'].apply(clean_text)
print(f"  Cleaned {len(merged)} tweets")
print(f"  Example: '{merged['Text_Tweet'].iloc[0][:80]}...'")
print(f"       →  '{merged['Text_Cleaned'].iloc[0][:80]}...'")

# ─── STEP 2: Case Folding ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Case Folding")
print("=" * 60)

merged['Text_CaseFolded'] = merged['Text_Cleaned'].str.lower()
print(f"  Converted all text to lowercase")
print(f"  Example: '{merged['Text_CaseFolded'].iloc[0][:80]}...'")

# ─── STEP 3: Tokenization ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Tokenization")
print("=" * 60)

merged['Tokens'] = merged['Text_CaseFolded'].apply(lambda x: x.split())
print(f"  Tokenized {len(merged)} tweets")
sample_tokens = merged['Tokens'].iloc[0][:10]
print(f"  Example tokens: {sample_tokens}")

avg_tokens = merged['Tokens'].apply(len).mean()
print(f"  Average tokens per tweet: {avg_tokens:.1f}")

# ─── STEP 4: Stopword Removal ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Stopword Removal (Sastrawi)")
print("=" * 60)

sw_factory = StopWordRemoverFactory()
stopwords = set(sw_factory.get_stop_words())
print(f"  Loaded {len(stopwords)} Indonesian stopwords")

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords and len(t) > 1]

merged['Tokens_NoStopwords'] = merged['Tokens'].apply(remove_stopwords)
avg_after = merged['Tokens_NoStopwords'].apply(len).mean()
print(f"  Average tokens before: {avg_tokens:.1f}, after: {avg_after:.1f}")
print(f"  Example: {merged['Tokens_NoStopwords'].iloc[0][:10]}")

# ─── STEP 5: Stemming ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Stemming (Sastrawi)")
print("=" * 60)

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

# Stem all tokens (this takes a while)
print("  Stemming in progress... (this may take a minute)")
merged['Tokens_Stemmed'] = merged['Tokens_NoStopwords'].apply(stem_tokens)

# Rejoin tokens into final preprocessed text
merged['Text_Preprocessed'] = merged['Tokens_Stemmed'].apply(lambda x: ' '.join(x))

print(f"  Stemming complete!")
print(f"  Example: {merged['Tokens_Stemmed'].iloc[0][:10]}")
print(f"  Final text: '{merged['Text_Preprocessed'].iloc[0][:80]}...'")

# ─── SUMMARY STATISTICS ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Total tweets: {len(merged)}")
print(f"  Unique keywords: {merged['Kategori_Keyword'].nunique()}")
print(f"  Categories: {merged['Kategori_Cyberbullying'].nunique()}")
print(f"\n  Category breakdown:")
for cat, count in merged['Kategori_Cyberbullying'].value_counts().items():
    pct = count / len(merged) * 100
    print(f"    {cat}: {count} ({pct:.1f}%)")

# ─── SAVE OUTPUT ─────────────────────────────────────────────────────────────────
print("\n  Saving output files...")

# Convert list columns to strings for Excel compatibility
output = merged.copy()
output['Tokens'] = output['Tokens'].apply(lambda x: json.dumps(x, ensure_ascii=False))
output['Tokens_NoStopwords'] = output['Tokens_NoStopwords'].apply(lambda x: json.dumps(x, ensure_ascii=False))
output['Tokens_Stemmed'] = output['Tokens_Stemmed'].apply(lambda x: json.dumps(x, ensure_ascii=False))

# Select and reorder columns
output = output[[
    'Kategori_Keyword',
    'Kategori_Cyberbullying',
    'Username',
    'Text_Tweet',
    'Text_Cleaned',
    'Text_CaseFolded',
    'Text_Preprocessed',
    'Tokens',
    'Tokens_NoStopwords',
    'Tokens_Stemmed'
]]

output.to_excel('/mnt/user-data/outputs/Merged_Preprocessed_Cyberbullying.xlsx', index=False, engine='openpyxl')
print("  Saved: Merged_Preprocessed_Cyberbullying.xlsx")

# Also save a clean CSV for ML pipeline
ml_output = merged[['Kategori_Cyberbullying', 'Text_Preprocessed', 'Text_Tweet']].copy()
ml_output.to_csv('/mnt/user-data/outputs/ML_Ready_Cyberbullying.csv', index=False, encoding='utf-8')
print("  Saved: ML_Ready_Cyberbullying.csv (ready for TF-IDF + classification)")

print("\n  Done!")
