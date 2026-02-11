import nltk

def ensure_nltk_data():
    packs = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
    ]
    for path, name in packs:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

ensure_nltk_data()

# Helper to build Grokipedia URL from a Wikipedia-style title
def make_grokipedia_url(title: str) -> str:
    """
    Build a Grokipedia URL from a Wikipedia-style title.
    If the page doesn't exist, the link will just 404 on click.
    """
    if not title:
        return ""
    slug = title.replace(" ", "_")
    return f"https://grokipedia.com/page/{slug}"

# ======================================================
# Hybrid Information Retrieval System (Wikipedia)
# Project: CSC-785-U18 Hybrid IR System
# ======================================================

import streamlit as st
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi, BM25L, BM25Plus   # <-- more BM25 variants
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# --- Initialize NLTK Resources ---
try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    st.error("NLTK data (wordnet, punkt, averaged_perceptron_tagger) is missing. Please run the download commands in your terminal.")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Hybrid IR System", layout="wide")
st.title("Hybrid Information Retrieval System (Wikipedia)")
st.caption("Blending Sparse (BM25) and Dense (SentenceTransformer) Retrieval with Lexical Query Expansion and Grokipedia comparison.")

# -----------------------------
# Sidebar: corpus + model settings
# -----------------------------
st.sidebar.header("Corpus Settings")

n_docs = st.sidebar.slider(
    "Number of Wikipedia articles",
    min_value=1000,
    max_value=20000,
    step=1000,
    value=10000,
    help="More documents = better coverage but slower startup and more RAM."
)

st.sidebar.header("Model Settings")

# Sparse (BM25) model selection
sparse_model_label = st.sidebar.selectbox(
    "Sparse model (BM25 variant)",
    ["BM25Okapi", "BM25L", "BM25Plus"],
    help="Different BM25 variants for lexical scoring."
)

# Dense (SentenceTransformer) model selection
dense_model_options = {
    "MiniLM-L6-v2 (fast, small)": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (better quality)": "sentence-transformers/all-mpnet-base-v2",
}
dense_model_label = st.sidebar.selectbox(
    "Dense model (SentenceTransformer)",
    list(dense_model_options.keys()),
    help="MiniLM is faster; MPNet is typically more accurate but slower."
)
dense_model_name = dense_model_options[dense_model_label]

# -----------------------------
# 1. Load Wikipedia Dataset
# -----------------------------
@st.cache_data(show_spinner="Loading and preprocessing Wikipedia corpus...")
def load_wikipedia_subset(n: int = 10000):
    """Loads n Wikipedia articles, returning text, titles, and URLs."""
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split=f"train[:{n}]"
        )

        docs, titles, urls = [], [], []
        for row in dataset:
            text = row["text"]
            if not isinstance(text, str):
                continue

            docs.append(text.lower().replace("\n", " ").strip())
            titles.append(row.get("title", "Untitled article"))
            urls.append(row.get("url", ""))
    except Exception as e:
        st.warning(
            f"⚠️ Couldn't load Wikipedia from Hugging Face. "
            f"Using local fallback. Error: {e}"
        )
        docs = [
            "Artificial intelligence simulates human intelligence processes by machines.",
            "Machine learning is a subset of AI focused on training models from data.",
            "Information retrieval is the process of obtaining relevant information from large repositories.",
        ]
        titles = ["AI", "ML", "IR"]
        urls = [""] * len(docs)

    return docs, titles, urls

documents, titles, urls = load_wikipedia_subset(n_docs)
st.sidebar.success(f"Loaded {len(documents)} documents for indexing.")

# -----------------------------
# 2. Initialize Models (BM25 + ST)
# -----------------------------
@st.cache_resource(show_spinner="Initializing BM25 and SentenceTransformer...")
def init_models(sparse_name: str, dense_name: str, docs):
    # tokenization for BM25
    tokenized_docs = [doc.split() for doc in docs]

    # choose BM25 variant
    if sparse_name == "BM25L":
        bm25 = BM25L(tokenized_docs)
    elif sparse_name == "BM25Plus":
        bm25 = BM25Plus(tokenized_docs)
    else:
        bm25 = BM25Okapi(tokenized_docs)

    # choose dense model
    model = SentenceTransformer(dense_name)

    embeddings = model.encode(
        docs,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    return bm25, model, embeddings

bm25, model, embeddings = init_models(sparse_model_label, dense_model_name, documents)
scaler = MinMaxScaler()

# -----------------------------
# 3. Query Expansion Logic
# -----------------------------
def get_wordnet_pos(treebank_tag):
    """Maps NLTK POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    return None

def apply_query_expansion(query):
    """Performs lexical query expansion using NLTK: Lemmatization + Synonyms."""
    tokens = nltk.word_tokenize(query)
    tagged_tokens = nltk.pos_tag(tokens)
    expanded_terms = set(tokens)  # Start with original tokens

    for word, tag in tagged_tokens:
        wn_tag = get_wordnet_pos(tag)

        # 1. Lemmatization
        if wn_tag:
            expanded_terms.add(lemmatizer.lemmatize(word, wn_tag))
        else:
            expanded_terms.add(lemmatizer.lemmatize(word))

        # 2. Synonym Expansion
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().lower().replace('_', ' ')
                expanded_terms.add(synonym)

    return " ".join(sorted(list(expanded_terms)))

# -----------------------------
# 4. Score Fusion Methods
# -----------------------------
def rrf_fusion(results_sparse, results_dense, k=60):
    """
    Reciprocal Rank Fusion (RRF) combines rankings, not scores.
    k is a smoothing constant, typically 60.
    """
    fused_scores = defaultdict(float)

    # Process Sparse Ranks
    for rank, doc_index in enumerate(results_sparse):
        fused_scores[doc_index] += 1.0 / (k + rank + 1)

    # Process Dense Ranks
    for rank, doc_index in enumerate(results_dense):
        fused_scores[doc_index] += 1.0 / (k + rank + 1)

    # Sort by fused score
    reranked_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return reranked_indices, fused_scores

# -----------------------------
# 5. Hybrid Search + Baselines
# -----------------------------
def hybrid_search(
    query,
    top_k=5,
    alpha=0.5,
    expand_query=True,
    fusion_method='Weighted Sum',
    retrieval_mode='hybrid'  # 'hybrid', 'bm25', 'dense'
):
    """
    Performs search with:
      - 'hybrid' : BM25 + Dense (Weighted Sum or RRF)
      - 'bm25'   : BM25 baseline only
      - 'dense'  : Dense baseline only
    """

    if not query.strip():
        return [], ""

    # Apply Query Expansion
    expanded_query = query
    if expand_query:
        expanded_query = apply_query_expansion(query)

    query_tokens = expanded_query.split()

    # --- Retrieve Scores ---
    CANDIDATE_K = 200

    # Sparse scores
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    sparse_top_indices = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]

    # Dense scores
    query_emb = model.encode(expanded_query, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
    dense_norm = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()
    dense_top_indices = np.argsort(dense_scores)[::-1][:CANDIDATE_K]

    final_scores = None  # used only for RRF

    # --- Decide ranking based on retrieval_mode ---
    if retrieval_mode == 'bm25':
        # BM25 baseline
        hybrid_score = bm25_norm
        top_indices = np.argsort(hybrid_score)[::-1][:top_k]

    elif retrieval_mode == 'dense':
        # Dense baseline
        hybrid_score = dense_norm
        top_indices = np.argsort(hybrid_score)[::-1][:top_k]

    else:
        # Hybrid (BM25 + Dense)
        if fusion_method == 'RRF (Reciprocal Rank Fusion)':
            final_indices, final_scores = rrf_fusion(sparse_top_indices, dense_top_indices)
            top_indices = final_indices[:top_k]
            hybrid_score = None  # not used directly
        else:
            # Weighted Sum (Linear Fusion)
            hybrid_score = alpha * dense_norm + (1 - alpha) * bm25_norm
            top_indices = np.argsort(hybrid_score)[::-1][:top_k]

    # 3. Compile Results
    results = []
    for idx in top_indices:
        if retrieval_mode == 'hybrid' and fusion_method == 'RRF (Reciprocal Rank Fusion)':
            primary_score = final_scores[idx]
        else:
            primary_score = hybrid_score[idx]

        results.append(
            {
                "text": documents[idx],
                "score": float(primary_score),
                "title": titles[idx],
                "url": urls[idx],
                "bm25_norm": float(bm25_norm[idx]),
                "dense_norm": float(dense_norm[idx]),
            }
        )
    return results, expanded_query

# -----------------------------
# 6. Streamlit UI
# -----------------------------
st.subheader("Query Configuration")
col_q, col_exp = st.columns([3, 1])

with col_q:
    query = st.text_input("🔍 Enter your query:", "What are the stages of information storage?")

with col_exp:
    expand_query = st.checkbox(
        '*Enable Query Expansion*',
        value=True,
        help="Uses NLTK (WordNet, Lemmatization) to broaden the query terms for better recall."
    )

st.markdown("---")
st.subheader("Fusion and Ranking Configuration")
col_mode, col_f, col_a, col_k = st.columns([2, 2, 1, 1])

with col_mode:
    mode_label = st.selectbox(
        "Retrieval Mode",
        [
            "Hybrid (BM25 + Dense)",
            "BM25 baseline (sparse only)",
            "Dense baseline (semantic only)"
        ],
        help="Compare hybrid retrieval with BM25-only and Dense-only baselines."
    )

if mode_label.startswith("Hybrid"):
    retrieval_mode = "hybrid"
elif "BM25" in mode_label:
    retrieval_mode = "bm25"
else:
    retrieval_mode = "dense"

with col_f:
    fusion_method = st.selectbox(
        "Hybrid Fusion Method",
        ['Weighted Sum (Linear)', 'RRF (Reciprocal Rank Fusion)'],
        help="Only used when retrieval mode is Hybrid."
    )

alpha_disabled = (fusion_method == 'RRF (Reciprocal Rank Fusion)') or (retrieval_mode != 'hybrid')

with col_a:
    alpha = st.slider(
        "Dense Weight (α)",
        0.0, 1.0, 0.5, 0.1,
        disabled=alpha_disabled,
        help="Balance between BM25 (0.0) and Dense (1.0) for Weighted Sum fusion."
    )
with col_k:
    top_k = st.slider("Top K Results", 3, 15, 5)

if st.button("Run Search", use_container_width=True):
    st.session_state.run_search = True

if st.session_state.get('run_search'):
    st.session_state.run_search = False

    with st.spinner(
        f"Searching for '{query}' using {mode_label} "
        f"({sparse_model_label} + {dense_model_label})..."
    ):
        results, expanded_query = hybrid_search(
            query,
            top_k=top_k,
            alpha=alpha,
            expand_query=expand_query,
            fusion_method=fusion_method,
            retrieval_mode=retrieval_mode
        )

    st.success(f"Successfully retrieved {len(results)} results.")

    if expand_query:
        st.info(f"Expanded Query used for indexing: *{expanded_query}*")

    st.subheader("Results")
    for i, res in enumerate(results, 1):
        st.markdown(f"### {i}. {res['title']}")

        score_cols = st.columns(3)

        if retrieval_mode == "hybrid" and fusion_method == 'RRF (Reciprocal Rank Fusion)':
            primary_label = "RRF Score (Rank-based)"
        elif retrieval_mode == "hybrid":
            primary_label = "Hybrid Score (Linear)"
        elif retrieval_mode == "bm25":
            primary_label = "BM25 baseline score"
        else:
            primary_label = "Dense baseline score"

        score_cols[0].metric(primary_label, f"{res['score']:.4f}")
        score_cols[1].metric("BM25 (Sparse) Norm", f"{res['bm25_norm']:.4f}")
        score_cols[2].metric("Dense (Vector) Norm", f"{res['dense_norm']:.4f}")

        # ----- Wikipedia + Grokipedia links -----
        wiki_url = res["url"]
        grok_url = make_grokipedia_url(res["title"])

        st.markdown(
            f"*Wikipedia URL:* {'[Link to Article]' if wiki_url else 'No link available'}"
        )

        link_cols = st.columns(2)
        with link_cols[0]:
            if wiki_url:
                st.link_button("Open Wikipedia article", wiki_url)
            else:
                st.write("No Wikipedia link")

        with link_cols[1]:
            st.link_button("Open Grokipedia (AI version)", grok_url)

        # Snippet
        st.markdown(f"> {res['text'][:700]}...")
        st.markdown("---")

with st.expander("About Wikipedia vs Grokipedia"):
    st.markdown(
        """
        - **Wikipedia**: human-edited, community-moderated encyclopedia with strict sourcing policies.  
        - **Grokipedia**: AI-generated encyclopedia powered by xAI's Grok model.  
        - This app **indexes Wikipedia only** for retrieval, but for each retrieved article
          we also provide a link to the corresponding **Grokipedia page**, so you can
          compare human-edited vs AI-generated content on the same topic.
        """
    )

st.caption(
    "Hybrid IR System — compare different BM25 variants and SentenceTransformer models, "
    "with query expansion, hybrid/baseline retrieval, and Wikipedia ↔ Grokipedia comparison."
)
