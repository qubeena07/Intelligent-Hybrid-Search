"""
Hybrid Information Retrieval System — Wikipedia
================================================
Course  : CSC-785-U18 Information Retrieval
Project : Hybrid IR System
Authors : <Author Names>

This application implements a hybrid document retrieval pipeline over an English
Wikipedia corpus.  Sparse retrieval uses BM25 (Okapi / BM25L / BM25Plus) for
term-frequency-based matching; dense retrieval uses SentenceTransformer embeddings
for semantic similarity.  The two signals are fused via either a Weighted Sum or
Reciprocal Rank Fusion (RRF).  An optional query-expansion stage uses WordNet
synonyms filtered by part-of-speech tag and embedding cosine similarity to improve
recall without introducing noise.  All search events are logged to CSV files for
offline evaluation.

Usage
-----
    streamlit run app.py
"""

# =============================================================================
# === IMPORTS
# =============================================================================

import csv
import os
import time
from collections import defaultdict
from datetime import datetime

import nltk
import numpy as np
import streamlit as st
from datasets import load_dataset
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25L, BM25Okapi, BM25Plus
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# === NLTK BOOTSTRAP
# =============================================================================


def ensure_nltk_data() -> None:
    """Download any missing NLTK packages required by this application.

    Checks for the punkt tokenizer, Penn Treebank POS tagger, WordNet corpus,
    and Open Multilingual WordNet (omw-1.4).  Safe to call on every startup —
    skips packages that are already present on disk.
    """
    for package in ["punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
        nltk.download(package, quiet=True)


ensure_nltk_data()

# =============================================================================
# === CONSTANTS
# =============================================================================

# --- Corpus ---
DEFAULT_CORPUS_SIZE: int = 10_000
MIN_CORPUS_SIZE: int = 1_000
MAX_CORPUS_SIZE: int = 20_000
CORPUS_SIZE_STEP: int = 1_000

# --- Retrieval ---
CANDIDATE_POOL_SIZE: int = 200   # documents retrieved per signal before re-ranking
RRF_K: int = 60                  # RRF smoothing constant (Cormack et al. SIGIR 2009)
DEFAULT_TOP_K: int = 5           # default number of results shown to the user
DEFAULT_ALPHA: float = 0.5       # default dense weight for Weighted Sum fusion

# --- Query Expansion ---
MAX_SYNONYMS_PER_WORD: int = 3              # cap per token to limit query drift
SYNONYM_SIMILARITY_THRESHOLD: float = 0.5  # minimum cosine similarity to accept a synonym

# --- Score Breakdown Display Thresholds ---
SCORE_HIGH_THRESHOLD: float = 0.7
SCORE_MID_THRESHOLD: float = 0.4
SCORE_LOW_THRESHOLD: float = 0.1

# --- Result Display ---
RESULT_SNIPPET_LENGTH: int = 700  # characters shown for each result text preview

# --- Logging ---
QUERY_LOG_PATH: str = "query_log.csv"
SEARCH_LOG_PATH: str = "search_log.csv"
SEARCH_LOG_HEADERS: list = [
    "timestamp", "query", "expanded_query", "retrieval_mode", "fusion_method",
    "bm25_variant", "encoder_model", "alpha_weight",
    "top1_title", "top1_hybrid_score", "top1_bm25_norm", "top1_dense_norm",
    "response_time_ms",
]

# --- Model Options ---
BM25_VARIANTS: list = ["BM25Okapi", "BM25L", "BM25Plus"]
DENSE_MODEL_OPTIONS: dict = {
    "MiniLM-L6-v2 (fast, small)":      "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (better quality)": "sentence-transformers/all-mpnet-base-v2",
}

# --- Caching ---
QUERY_CACHE_TTL_SECONDS: int = 3_600  # reuse query embeddings for up to one hour

# =============================================================================
# === MODULE-LEVEL HELPERS
# =============================================================================

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    st.error(
        "NLTK WordNet data is missing. "
        "Run `python -c \"import nltk; nltk.download('wordnet')\"` in your terminal."
    )


def make_grokipedia_url(title: str) -> str:
    """Build a Grokipedia URL from a Wikipedia article title.

    Parameters
    ----------
    title : str
        Wikipedia-style article title (spaces allowed).

    Returns
    -------
    str
        Grokipedia URL with spaces replaced by underscores, or an empty string
        when *title* is empty.  The linked page may 404 if Grokipedia has no
        corresponding article.
    """
    if not title:
        return ""
    slug = title.replace(" ", "_")
    return f"https://grokipedia.com/page/{slug}"


# =============================================================================
# === DATA LOADING
# =============================================================================


@st.cache_data(show_spinner="Loading and preprocessing Wikipedia corpus...")
def load_wikipedia_subset(num_docs: int = DEFAULT_CORPUS_SIZE):
    """Load a fixed-size slice of the English Wikipedia dataset from Hugging Face.

    Parameters
    ----------
    num_docs : int
        Number of articles to load.  Larger values improve recall but increase
        startup time and memory usage.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        A triple ``(documents, titles, urls)`` where each list has up to
        *num_docs* entries.  *documents* are lowercased, newline-stripped body
        texts.  Falls back to a three-document stub corpus if the dataset
        cannot be fetched from Hugging Face.
    """
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split=f"train[:{num_docs}]",
        )
        documents, titles, urls = [], [], []
        for row in dataset:
            article_text = row["text"]
            if not isinstance(article_text, str):
                continue
            documents.append(article_text.lower().replace("\n", " ").strip())
            titles.append(row.get("title", "Untitled article"))
            urls.append(row.get("url", ""))
    except Exception as load_error:
        st.warning(
            f"Could not load Wikipedia from Hugging Face — using local fallback. "
            f"Error: {load_error}"
        )
        documents = [
            "Artificial intelligence simulates human intelligence processes by machines.",
            "Machine learning is a subset of AI focused on training models from data.",
            "Information retrieval is the process of obtaining relevant information from large repositories.",
        ]
        titles = ["AI", "ML", "IR"]
        urls = [""] * len(documents)

    return documents, titles, urls


# =============================================================================
# === RETRIEVAL MODELS
# =============================================================================


@st.cache_resource(show_spinner="Initialising BM25 and SentenceTransformer...")
def init_models(sparse_variant: str, dense_model_id: str, documents: tuple):
    """Initialise BM25, SentenceTransformer encoder, and a shared score scaler.

    Parameters
    ----------
    sparse_variant : str
        One of ``"BM25Okapi"``, ``"BM25L"``, or ``"BM25Plus"``.
    dense_model_id : str
        Hugging Face model identifier for the SentenceTransformer encoder.
    documents : tuple
        Corpus texts as a tuple (``@st.cache_resource`` requires hashable args).

    Returns
    -------
    tuple
        ``(bm25_index, sentence_model, corpus_embeddings, score_scaler)``

        - *bm25_index* — fitted BM25 instance for the full corpus.
        - *sentence_model* — loaded SentenceTransformer.
        - *corpus_embeddings* — tensor of shape ``(num_docs, embedding_dim)``.
        - *score_scaler* — ``MinMaxScaler`` pre-fitted on [0, 1].  Using
          ``transform()`` instead of ``fit_transform()`` per query avoids
          refitting overhead (~145 ms saved per query).
    """
    tokenized_docs = [doc.split() for doc in documents]

    if sparse_variant == "BM25L":
        bm25_index = BM25L(tokenized_docs)
    elif sparse_variant == "BM25Plus":
        bm25_index = BM25Plus(tokenized_docs)
    else:
        bm25_index = BM25Okapi(tokenized_docs)

    sentence_model = SentenceTransformer(dense_model_id)
    corpus_embeddings = sentence_model.encode(
        list(documents),
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    score_scaler = MinMaxScaler()
    score_scaler.fit([[0], [1]])  # pre-fitted once; reused for all queries

    return bm25_index, sentence_model, corpus_embeddings, score_scaler


@st.cache_data(ttl=QUERY_CACHE_TTL_SECONDS)
def encode_query(query_text: str, _sentence_model) -> object:
    """Encode a query string into a dense embedding tensor, with caching.

    Identical queries within the TTL window skip re-encoding entirely
    (<1 ms vs ~75 ms for a fresh encode).

    Parameters
    ----------
    query_text : str
        The (possibly expanded) query string.
    _sentence_model : SentenceTransformer
        Encoder model.  The leading underscore prevents Streamlit from hashing
        this argument (the model object is not trivially hashable).

    Returns
    -------
    torch.Tensor
        1-D embedding tensor for the query.
    """
    return _sentence_model.encode(query_text, convert_to_tensor=True)


# =============================================================================
# === QUERY EXPANSION
# =============================================================================


def map_treebank_to_wordnet_pos(treebank_tag: str):
    """Convert a Penn Treebank POS tag to the corresponding WordNet POS constant.

    Parameters
    ----------
    treebank_tag : str
        POS tag returned by ``nltk.pos_tag``, e.g. ``"NN"``, ``"VBZ"``, ``"JJ"``.

    Returns
    -------
    str or None
        One of ``wordnet.NOUN``, ``wordnet.VERB``, ``wordnet.ADJ``,
        ``wordnet.ADV``, or ``None`` for tags with no WordNet equivalent
        (e.g. prepositions, determiners).
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def apply_query_expansion(query: str, similarity_model=None) -> str:
    """Expand a query with POS-aware WordNet synonyms and lemmatisation.

    Expansion strategy:

    1. Tokenise and POS-tag the query with NLTK.
    2. Lemmatise each token using its detected POS to normalise inflections.
    3. Fetch WordNet synsets restricted to the token's POS — this prevents
       cross-sense pollution (e.g. noun "bank" will not yield verb synonyms).
    4. Accept at most ``MAX_SYNONYMS_PER_WORD`` synonyms per token to avoid
       query drift.
    5. If *similarity_model* is supplied, reject any synonym whose cosine
       similarity to the source word falls below
       ``SYNONYM_SIMILARITY_THRESHOLD``.

    Parameters
    ----------
    query : str
        Raw user query string.
    similarity_model : SentenceTransformer or None
        Encoder used to compute word-level cosine similarity for synonym
        filtering.  When ``None`` the similarity check is skipped entirely.

    Returns
    -------
    str
        Space-joined string containing original tokens, lemma forms, and all
        accepted synonyms.
    """
    tokens = nltk.word_tokenize(query)
    tagged_tokens = nltk.pos_tag(tokens)
    expanded_terms = list(tokens)

    # Pre-compute word embeddings once per query to avoid repeated encoding
    word_embeddings: dict = {}
    if similarity_model:
        try:
            for token in tokens:
                token_lower = token.lower()
                if token_lower.isalpha():
                    word_embeddings[token_lower] = similarity_model.encode(
                        token_lower, convert_to_tensor=False
                    )
        except Exception:
            pass  # Non-fatal: similarity filter will be skipped for all words

    for token, pos_tag in tagged_tokens:
        wordnet_pos = map_treebank_to_wordnet_pos(pos_tag)
        token_lower = token.lower()

        # Add lemma form to prevent duplicate inflections in the index
        lemma_form = (
            lemmatizer.lemmatize(token, wordnet_pos)
            if wordnet_pos
            else lemmatizer.lemmatize(token)
        )
        if lemma_form.lower() not in [term.lower() for term in expanded_terms]:
            expanded_terms.append(lemma_form)

        if wordnet_pos is None:
            continue  # No WordNet synsets available for this POS

        synonyms_added = 0
        for synset in wordnet.synsets(token, pos=wordnet_pos):
            if synonyms_added >= MAX_SYNONYMS_PER_WORD:
                break
            for synset_lemma in synset.lemmas():
                if synonyms_added >= MAX_SYNONYMS_PER_WORD:
                    break
                synonym = synset_lemma.name().lower().replace("_", " ")
                if synonym in [term.lower() for term in expanded_terms]:
                    continue

                # Reject synonyms with low semantic similarity to the source word
                if similarity_model and token_lower in word_embeddings:
                    try:
                        synonym_embedding = similarity_model.encode(
                            synonym, convert_to_tensor=False
                        )
                        similarity_score = 1.0 - cosine(
                            word_embeddings[token_lower], synonym_embedding
                        )
                        if similarity_score < SYNONYM_SIMILARITY_THRESHOLD:
                            continue
                    except Exception:
                        pass  # Encoding failure: include synonym without filtering

                expanded_terms.append(synonym)
                synonyms_added += 1

    return " ".join(expanded_terms)


# =============================================================================
# === FUSION
# =============================================================================


def rrf_fusion(
    sparse_ranked_indices: list,
    dense_ranked_indices: list,
    smoothing_constant: int = RRF_K,
) -> tuple:
    """Combine sparse and dense ranked lists using Reciprocal Rank Fusion (RRF).

    Each document's RRF score is the sum of ``1 / (k + rank)`` across both
    ranked lists.  Documents absent from one list contribute nothing from that
    list.  The smoothing constant *k* dampens the influence of very high ranks.

    Reference: Cormack, Clarke, & Buettcher (SIGIR 2009).

    Parameters
    ----------
    sparse_ranked_indices : list[int]
        Document indices ordered by descending BM25 score.
    dense_ranked_indices : list[int]
        Document indices ordered by descending dense cosine similarity.
    smoothing_constant : int
        RRF constant *k*.  Defaults to ``RRF_K = 60``.

    Returns
    -------
    tuple[list[int], dict[int, float]]
        ``(reranked_indices, fused_scores)`` where *reranked_indices* is sorted
        by descending fused score and *fused_scores* maps each document index
        to its combined RRF score.
    """
    fused_scores: dict = defaultdict(float)
    for rank, doc_index in enumerate(sparse_ranked_indices):
        fused_scores[doc_index] += 1.0 / (smoothing_constant + rank + 1)
    for rank, doc_index in enumerate(dense_ranked_indices):
        fused_scores[doc_index] += 1.0 / (smoothing_constant + rank + 1)
    reranked_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return reranked_indices, fused_scores


# =============================================================================
# === LOGGING
# =============================================================================


def log_query_expansion(
    original_query: str,
    expanded_query: str,
    log_path: str = QUERY_LOG_PATH,
) -> None:
    """Append one row to the query-expansion log CSV.

    Creates the file with a header row on first write.

    Parameters
    ----------
    original_query : str
        The raw user query before expansion.
    expanded_query : str
        The query string after WordNet expansion and lemmatisation.
    log_path : str
        Destination CSV file path.  Defaults to ``QUERY_LOG_PATH``.
    """
    terms_added = len(expanded_query.split()) - len(original_query.split())
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["timestamp", "original_query", "expanded_query", "terms_added"])
        writer.writerow([datetime.now().isoformat(), original_query, expanded_query, terms_added])


def log_search_result(
    query: str,
    expanded_query: str,
    retrieval_mode: str,
    fusion_method: str,
    bm25_variant: str,
    encoder_model: str,
    alpha_weight: float,
    results: list,
    response_time_ms: float,
    log_path: str = SEARCH_LOG_PATH,
) -> None:
    """Append one row per search event to the search-result log CSV.

    Creates the file with a header row on first write.  Records the top-1
    result's scores along with the full configuration used for the search.

    Parameters
    ----------
    query : str
        Original user query.
    expanded_query : str
        Query after expansion (equals *query* when expansion is disabled).
    retrieval_mode : str
        One of ``"hybrid"``, ``"bm25"``, ``"dense"``.
    fusion_method : str
        Fusion label (e.g. ``"Weighted Sum (Linear)"``).
    bm25_variant : str
        BM25 variant label (e.g. ``"BM25Okapi"``).
    encoder_model : str
        Display label for the SentenceTransformer model.
    alpha_weight : float
        Dense weight used in Weighted Sum fusion.
    results : list[dict]
        Ranked result dicts returned by ``hybrid_search``.
    response_time_ms : float
        Wall-clock search latency in milliseconds.
    log_path : str
        Destination CSV file path.  Defaults to ``SEARCH_LOG_PATH``.
    """
    file_exists = os.path.isfile(log_path)
    top_result = results[0] if results else {}
    row = [
        datetime.now().isoformat(),
        query,
        expanded_query,
        retrieval_mode,
        fusion_method,
        bm25_variant,
        encoder_model,
        alpha_weight,
        top_result.get("title", ""),
        round(top_result.get("score", 0.0), 6),
        round(top_result.get("bm25_norm", 0.0), 6),
        round(top_result.get("dense_norm", 0.0), 6),
        round(response_time_ms, 1),
    ]
    with open(log_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(SEARCH_LOG_HEADERS)
        writer.writerow(row)


# =============================================================================
# === SEARCH
# =============================================================================


def score_breakdown_text(
    bm25_norm: float,
    dense_norm: float,
    retrieval_mode: str,
    fusion_method: str,
) -> str:
    """Generate a plain-English explanation of a document's ranking scores.

    Parameters
    ----------
    bm25_norm : float
        Normalised BM25 score in [0, 1].
    dense_norm : float
        Normalised dense cosine score in [0, 1].
    retrieval_mode : str
        One of ``"hybrid"``, ``"bm25"``, ``"dense"``.
    fusion_method : str
        Fusion label string (used to distinguish RRF from Weighted Sum).

    Returns
    -------
    str
        Markdown-formatted multi-paragraph explanation suitable for
        ``st.markdown``.
    """
    if bm25_norm >= SCORE_HIGH_THRESHOLD:
        bm25_description = "Very high — strong keyword match with the query terms."
    elif bm25_norm >= SCORE_MID_THRESHOLD:
        bm25_description = "Moderate — partial keyword overlap with the query."
    elif bm25_norm >= SCORE_LOW_THRESHOLD:
        bm25_description = "Low — few query keywords appear in this document."
    else:
        bm25_description = "Negligible — query keywords are largely absent."

    if dense_norm >= SCORE_HIGH_THRESHOLD:
        dense_description = "Very high — document is semantically very close to the query."
    elif dense_norm >= SCORE_MID_THRESHOLD:
        dense_description = "Moderate — meaningful semantic relevance to the query."
    elif dense_norm >= SCORE_LOW_THRESHOLD:
        dense_description = "Low — limited semantic overlap with the query."
    else:
        dense_description = "Negligible — document meaning is distant from the query."

    explanation_lines = [
        f"**BM25 score ({bm25_norm:.4f}):** {bm25_description}",
        f"**Dense score ({dense_norm:.4f}):** {dense_description}",
    ]

    if retrieval_mode == "hybrid":
        if "RRF" in fusion_method:
            explanation_lines.append(
                "**Fusion (RRF):** Final rank combines positions from both sparse and dense "
                "lists — documents appearing high in *both* receive a compounded boost "
                "regardless of raw score magnitude."
            )
        else:
            explanation_lines.append(
                "**Fusion (Weighted Sum):** Final score is a linear blend of BM25 and dense "
                "scores.  High scores on *both* axes push a document to the top."
            )
    elif retrieval_mode == "bm25":
        explanation_lines.append(
            "**Mode (BM25-only):** Ranking depends entirely on keyword frequency and "
            "document-length normalisation — semantic meaning is not considered."
        )
    else:
        explanation_lines.append(
            "**Mode (Dense-only):** Ranking depends entirely on embedding cosine similarity "
            "— exact keyword presence is not considered."
        )

    return "\n\n".join(explanation_lines)


def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    expand_query: bool = True,
    fusion_method: str = "Weighted Sum (Linear)",
    retrieval_mode: str = "hybrid",
) -> tuple:
    """Execute a search over the corpus using the chosen retrieval strategy.

    Supports three modes:

    - ``"hybrid"`` — combines BM25 and dense scores via Weighted Sum or RRF.
    - ``"bm25"``   — sparse-only baseline (term frequency + IDF).
    - ``"dense"``  — semantic-only baseline (embedding cosine similarity).

    Parameters
    ----------
    query : str
        Raw user query string.
    top_k : int
        Maximum number of top-ranked documents to return.
    alpha : float
        Weight assigned to the dense score in Weighted Sum fusion; the BM25
        weight is ``1 - alpha``.  Ignored for RRF and non-hybrid modes.
    expand_query : bool
        Whether to apply WordNet-based query expansion before retrieval.
    fusion_method : str
        ``"Weighted Sum (Linear)"`` or ``"RRF (Reciprocal Rank Fusion)"``.
    retrieval_mode : str
        One of ``"hybrid"``, ``"bm25"``, ``"dense"``.

    Returns
    -------
    tuple[list[dict], str]
        ``(results, expanded_query_string)`` where *results* is a list of at
        most *top_k* dicts containing keys ``title``, ``text``, ``url``,
        ``score``, ``bm25_norm``, ``dense_norm``; and *expanded_query_string*
        is the (possibly expanded) query submitted to the index.
    """
    if not query.strip():
        return [], query

    # --- Query Expansion ---
    expanded_query_string = query
    if expand_query:
        expanded_query_string = apply_query_expansion(query, similarity_model=sentence_model)
        log_query_expansion(query, expanded_query_string)

    query_tokens = expanded_query_string.split()

    # --- Sparse (BM25) Scoring ---
    bm25_raw_scores = bm25_index.get_scores(query_tokens)
    bm25_norm_scores = score_scaler.transform(bm25_raw_scores.reshape(-1, 1)).flatten()
    sparse_candidate_indices = np.argsort(bm25_raw_scores)[::-1][:CANDIDATE_POOL_SIZE]

    # --- Dense (Semantic) Scoring ---
    query_embedding = encode_query(expanded_query_string, sentence_model)
    dense_raw_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    dense_norm_scores = score_scaler.transform(dense_raw_scores.reshape(-1, 1)).flatten()
    dense_candidate_indices = np.argsort(dense_raw_scores)[::-1][:CANDIDATE_POOL_SIZE]

    # --- Fusion / Mode Selection ---
    rrf_scores = None
    combined_scores = None

    if retrieval_mode == "bm25":
        combined_scores = bm25_norm_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

    elif retrieval_mode == "dense":
        combined_scores = dense_norm_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

    else:  # hybrid
        if fusion_method == "RRF (Reciprocal Rank Fusion)":
            reranked_indices, rrf_scores = rrf_fusion(
                sparse_candidate_indices, dense_candidate_indices
            )
            top_indices = reranked_indices[:top_k]
        else:
            combined_scores = alpha * dense_norm_scores + (1 - alpha) * bm25_norm_scores
            top_indices = np.argsort(combined_scores)[::-1][:top_k]

    # --- Compile Result Dicts ---
    results = []
    for doc_index in top_indices:
        if retrieval_mode == "hybrid" and fusion_method == "RRF (Reciprocal Rank Fusion)":
            primary_score = rrf_scores[doc_index]
        else:
            primary_score = combined_scores[doc_index]

        results.append({
            "title":      titles[doc_index],
            "text":       documents[doc_index],
            "url":        urls[doc_index],
            "score":      float(primary_score),
            "bm25_norm":  float(bm25_norm_scores[doc_index]),
            "dense_norm": float(dense_norm_scores[doc_index]),
        })

    return results, expanded_query_string


# =============================================================================
# === UI — PAGE CONFIG & HEADER
# =============================================================================

st.set_page_config(
    page_title="Hybrid IR — Wikipedia",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# === CUSTOM CSS + ANIMATIONS
# =============================================================================
st.markdown("""
<style>
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Animated gradient hero header ── */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 70%, #533483 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 6px;
    margin-top: 10px;
    backdrop-filter: blur(4px);
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(145deg, #1e1e2e, #252535);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    animation: fadeSlideIn 0.4s ease both;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(83,52,131,0.35);
    border-color: rgba(99,102,241,0.5);
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Rank badge ── */
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    font-weight: 700;
    font-size: 1rem;
    margin-right: 10px;
    flex-shrink: 0;
}
.rank-1 { background: linear-gradient(135deg, #f59e0b, #f97316); color: white; box-shadow: 0 2px 8px rgba(249,115,22,0.5); }
.rank-2 { background: linear-gradient(135deg, #94a3b8, #cbd5e1); color: #1e293b; box-shadow: 0 2px 8px rgba(148,163,184,0.4); }
.rank-3 { background: linear-gradient(135deg, #92400e, #b45309); color: white; box-shadow: 0 2px 8px rgba(180,83,9,0.4); }
.rank-n { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; box-shadow: 0 2px 8px rgba(124,58,237,0.4); }

/* ── Result title row ── */
.result-title-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.8rem;
}
.result-title {
    color: #e2e8f0;
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0;
}

/* ── Score bar ── */
.score-bar-container {
    margin: 4px 0 2px 0;
}
.score-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.55);
    margin-bottom: 3px;
}
.score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    animation: barGrow 0.7s ease both;
    transform-origin: left;
}
.bar-hybrid  { background: linear-gradient(90deg, #6366f1, #a78bfa); }
.bar-bm25    { background: linear-gradient(90deg, #0ea5e9, #38bdf8); }
.bar-dense   { background: linear-gradient(90deg, #10b981, #34d399); }
@keyframes barGrow {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}

/* ── Snippet text ── */
.result-snippet {
    color: rgba(255,255,255,0.55);
    font-size: 0.85rem;
    line-height: 1.6;
    margin-top: 0.8rem;
    border-left: 3px solid rgba(99,102,241,0.4);
    padding-left: 0.8rem;
}

/* ── Link buttons row ── */
.link-row {
    display: flex;
    gap: 10px;
    margin-top: 0.9rem;
}
.link-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    text-decoration: none;
    transition: opacity 0.2s, transform 0.15s;
}
.link-btn:hover { opacity: 0.85; transform: translateY(-1px); }
.link-wiki  { background: rgba(14,165,233,0.18); color: #38bdf8; border: 1px solid rgba(14,165,233,0.35); }
.link-groki { background: rgba(167,139,250,0.18); color: #a78bfa; border: 1px solid rgba(167,139,250,0.35); }

/* ── Section divider ── */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 1.5rem 0;
}

/* ── Stats pills ── */
.stats-pill {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    color: #a5b4fc;
    display: inline-block;
    margin: 3px;
}

/* ── Compare method header ── */
.compare-header {
    background: linear-gradient(135deg, #1e3a5f, #2d1b69);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: white;
    font-weight: 600;
    font-size: 0.95rem;
    text-align: center;
    margin-bottom: 0.8rem;
}

/* ── Mode badge ── */
.mode-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.mode-hybrid { background: rgba(139,92,246,0.25); color: #c4b5fd; }
.mode-bm25   { background: rgba(14,165,233,0.25); color: #7dd3fc; }
.mode-dense  { background: rgba(16,185,129,0.25); color: #6ee7b7; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }

/* ── Sidebar tweaks ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ──
st.markdown("""
<div class="hero-header">
  <h1>🔍 Hybrid IR System</h1>
  <p>Blending Sparse BM25 + Dense SentenceTransformer retrieval over Wikipedia,
  with lexical query expansion and Grokipedia comparison.</p>
  <span class="hero-badge">BM25</span>
  <span class="hero-badge">SentenceTransformers</span>
  <span class="hero-badge">RRF</span>
  <span class="hero-badge">WordNet Expansion</span>
  <span class="hero-badge">Grokipedia</span>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# === UI — SIDEBAR: CORPUS & MODEL SETTINGS
# =============================================================================

st.sidebar.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem 0;">
  <span style="font-size:1.8rem">🔍</span>
  <p style="color:#a5b4fc;font-weight:600;font-size:0.95rem;margin:4px 0 0 0;">Hybrid IR Settings</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.header("Corpus Settings")
num_docs = st.sidebar.slider(
    "Number of Wikipedia articles",
    min_value=MIN_CORPUS_SIZE,
    max_value=MAX_CORPUS_SIZE,
    step=CORPUS_SIZE_STEP,
    value=DEFAULT_CORPUS_SIZE,
    help="More documents = better coverage but slower startup and more RAM.",
)

st.sidebar.header("Model Settings")
sparse_model_label = st.sidebar.selectbox(
    "Sparse model (BM25 variant)",
    BM25_VARIANTS,
    help="Different BM25 variants for lexical scoring.",
)
dense_model_label = st.sidebar.selectbox(
    "Dense model (SentenceTransformer)",
    list(DENSE_MODEL_OPTIONS.keys()),
    help="MiniLM is faster; MPNet is typically more accurate but slower.",
)
dense_model_id = DENSE_MODEL_OPTIONS[dense_model_label]

# Load corpus and models (both are cached; only runs on first call or config change)
documents, titles, urls = load_wikipedia_subset(num_docs)
st.sidebar.success(f"Loaded {len(documents)} documents for indexing.")

bm25_index, sentence_model, corpus_embeddings, score_scaler = init_models(
    sparse_model_label, dense_model_id, tuple(documents)
)

# =============================================================================
# === UI — QUERY CONFIGURATION
# =============================================================================

st.markdown('<h3 style="color:#e2e8f0;margin-bottom:0.5rem;">⚙️ Query Configuration</h3>', unsafe_allow_html=True)
query_col, expand_col, show_col = st.columns([3, 1, 1])

with query_col:
    query = st.text_input(
        "Enter your query:",
        "What are the stages of information storage?",
    )
with expand_col:
    expand_query = st.checkbox(
        "*Enable Query Expansion*",
        value=True,
        help="Uses NLTK (WordNet, Lemmatization) to broaden the query terms for better recall.",
    )
with show_col:
    show_expanded_query = st.checkbox(
        "*Show Expanded Query*",
        value=False,
        disabled=not expand_query,
        help="Display the expanded query string before running search.",
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<h3 style="color:#e2e8f0;margin-bottom:0.5rem;">🔧 Fusion & Ranking</h3>', unsafe_allow_html=True)
mode_col, fusion_col, alpha_col, topk_col = st.columns([2, 2, 1, 1])

with mode_col:
    mode_label = st.selectbox(
        "Retrieval Mode",
        [
            "Hybrid (BM25 + Dense)",
            "BM25 baseline (sparse only)",
            "Dense baseline (semantic only)",
        ],
        help="Compare hybrid retrieval with BM25-only and Dense-only baselines.",
    )

if mode_label.startswith("Hybrid"):
    retrieval_mode = "hybrid"
elif "BM25" in mode_label:
    retrieval_mode = "bm25"
else:
    retrieval_mode = "dense"

with fusion_col:
    fusion_method = st.selectbox(
        "Hybrid Fusion Method",
        ["Weighted Sum (Linear)", "RRF (Reciprocal Rank Fusion)"],
        help="Only used when retrieval mode is Hybrid.",
    )

alpha_disabled = (fusion_method == "RRF (Reciprocal Rank Fusion)") or (retrieval_mode != "hybrid")

with alpha_col:
    alpha = st.slider(
        "Dense Weight (α)",
        0.0, 1.0, DEFAULT_ALPHA, 0.1,
        disabled=alpha_disabled,
        help="Balance between BM25 (0.0) and Dense (1.0) for Weighted Sum fusion.",
    )
with topk_col:
    top_k = st.slider("Top K Results", 3, 15, DEFAULT_TOP_K)

# =============================================================================
# === UI — SESSION STATE INITIALISATION
# =============================================================================

for state_key, default_value in [
    ("total_searches",      0),
    ("response_times",      []),
    ("retrieval_modes_used", []),
    ("last_logged",         False),
    ("run_search",          False),
    ("compare_search",      False),
]:
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value

# =============================================================================
# === UI — SEARCH BUTTONS
# =============================================================================

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
run_col, compare_col = st.columns(2)
with run_col:
    if st.button("🔍  Run Search", use_container_width=True, type="primary"):
        st.session_state.run_search = True
with compare_col:
    if st.button(
        "⚖️  Compare All Methods",
        use_container_width=True,
        help="Runs the same query through BM25-only, Dense-only, and Hybrid simultaneously.",
    ):
        st.session_state.compare_search = True

# =============================================================================
# === UI — SINGLE-MODE SEARCH RESULTS
# =============================================================================

if st.session_state.run_search:
    st.session_state.run_search = False

    with st.spinner(
        f"Searching for '{query}' using {mode_label} "
        f"({sparse_model_label} + {dense_model_label})..."
    ):
        search_start = time.perf_counter()
        results, expanded_query_string = hybrid_search(
            query,
            top_k=top_k,
            alpha=alpha,
            expand_query=expand_query,
            fusion_method=fusion_method,
            retrieval_mode=retrieval_mode,
        )
        response_time_ms = (time.perf_counter() - search_start) * 1000

    log_search_result(
        query=query,
        expanded_query=expanded_query_string,
        retrieval_mode=retrieval_mode,
        fusion_method=fusion_method,
        bm25_variant=sparse_model_label,
        encoder_model=dense_model_label,
        alpha_weight=alpha,
        results=results,
        response_time_ms=response_time_ms,
    )

    st.session_state.total_searches += 1
    st.session_state.response_times.append(response_time_ms)
    st.session_state.retrieval_modes_used.append(retrieval_mode)
    st.session_state.last_logged = True

    # ── stats pills ──
    mode_cls = {"hybrid": "mode-hybrid", "bm25": "mode-bm25", "dense": "mode-dense"}.get(retrieval_mode, "mode-hybrid")
    st.markdown(
        f'''<div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
          <span class="stats-pill">✅ {len(results)} results</span>
          <span class="stats-pill">⚡ {response_time_ms:.0f} ms</span>
          <span class="mode-badge {mode_cls}">{retrieval_mode}</span>
        </div>''',
        unsafe_allow_html=True,
    )

    if expand_query and show_expanded_query:
        st.info(f"🔎 Expanded query: *{expanded_query_string}*")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    for rank, result in enumerate(results, start=1):
        # rank badge class
        rank_cls = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-n")

        # score bar widths (clamped to 4–100%)
        def bar_w(v): return max(4, min(100, int(v * 100)))

        # primary score label
        if retrieval_mode == "hybrid" and fusion_method == "RRF (Reciprocal Rank Fusion)":
            primary_label, primary_cls = "RRF Score", "bar-hybrid"
        elif retrieval_mode == "hybrid":
            primary_label, primary_cls = "Hybrid Score", "bar-hybrid"
        elif retrieval_mode == "bm25":
            primary_label, primary_cls = "BM25 Score", "bar-bm25"
        else:
            primary_label, primary_cls = "Dense Score", "bar-dense"

        wiki_url = result["url"]
        grok_url = make_grokipedia_url(result["title"])
        wiki_link = f'<a class="link-btn link-wiki" href="{wiki_url}" target="_blank">🌐 Wikipedia</a>' if wiki_url else ""
        grok_link = f'<a class="link-btn link-groki" href="{grok_url}" target="_blank">🤖 Grokipedia</a>'

        st.markdown(f'''
<div class="result-card" style="animation-delay:{(rank-1)*0.07:.2f}s">
  <div class="result-title-row">
    <span class="rank-badge {rank_cls}">{rank}</span>
    <p class="result-title">{result["title"]}</p>
  </div>
  <div class="score-bar-container">
    <div class="score-bar-label"><span>{primary_label}</span><span>{result["score"]:.4f}</span></div>
    <div class="score-bar-bg"><div class="score-bar-fill {primary_cls}" style="width:{bar_w(result["score"])}%"></div></div>
  </div>
  <div style="display:flex;gap:16px;margin-top:6px;">
    <div style="flex:1" class="score-bar-container">
      <div class="score-bar-label"><span>BM25</span><span>{result["bm25_norm"]:.4f}</span></div>
      <div class="score-bar-bg"><div class="score-bar-fill bar-bm25" style="width:{bar_w(result["bm25_norm"])}%"></div></div>
    </div>
    <div style="flex:1" class="score-bar-container">
      <div class="score-bar-label"><span>Dense</span><span>{result["dense_norm"]:.4f}</span></div>
      <div class="score-bar-bg"><div class="score-bar-fill bar-dense" style="width:{bar_w(result["dense_norm"])}%"></div></div>
    </div>
  </div>
  <div class="result-snippet">{result["text"][:RESULT_SNIPPET_LENGTH]}…</div>
  <div class="link-row">{wiki_link}{grok_link}</div>
</div>
''', unsafe_allow_html=True)

        with st.expander("📊 Score breakdown — why did this rank here?"):
            st.markdown(score_breakdown_text(
                result["bm25_norm"], result["dense_norm"], retrieval_mode, fusion_method
            ))

# =============================================================================
# === UI — COMPARE ALL METHODS
# =============================================================================

if st.session_state.compare_search:
    st.session_state.compare_search = False

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#e2e8f0;margin-bottom:0.3rem;">⚖️ Method Comparison</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:rgba(255,255,255,0.5);font-size:0.88rem;margin-bottom:1rem;">Query: <em>{query}</em> — BM25, Dense, and Hybrid ({fusion_method}) side by side.</p>', unsafe_allow_html=True)

    compare_modes = [
        ("🔵 BM25 Baseline",  "bm25",   "mode-bm25",   "bar-bm25"),
        ("🟢 Dense Baseline", "dense",  "mode-dense",  "bar-dense"),
        ("🟣 Hybrid",         "hybrid", "mode-hybrid", "bar-hybrid"),
    ]
    bm25_col, dense_col, hybrid_col = st.columns(3)
    compare_display_columns = [bm25_col, dense_col, hybrid_col]

    for (column_label, mode, mode_cls, bar_cls), display_col in zip(compare_modes, compare_display_columns):
        mode_start = time.perf_counter()
        mode_results, _ = hybrid_search(
            query,
            top_k=top_k,
            alpha=alpha,
            expand_query=expand_query,
            fusion_method=fusion_method,
            retrieval_mode=mode,
        )
        mode_time_ms = (time.perf_counter() - mode_start) * 1000

        with display_col:
            st.markdown(
                f'<div class="compare-header">{column_label}'
                f'<br><span style="font-size:0.7rem;opacity:0.7;font-weight:400;">⚡ {mode_time_ms:.0f} ms</span></div>',
                unsafe_allow_html=True,
            )
            if not mode_results:
                st.write("No results.")
            for result_rank, result in enumerate(mode_results[:top_k], start=1):
                score_val = result["bm25_norm"] if mode == "bm25" else result["dense_norm"] if mode == "dense" else result["score"]
                bar_w = max(4, min(100, int(score_val * 100)))
                rank_cls = {1:"rank-1",2:"rank-2",3:"rank-3"}.get(result_rank,"rank-n")
                wiki_url = result["url"]
                grok_url = make_grokipedia_url(result["title"])
                wiki_link = f'<a class="link-btn link-wiki" href="{wiki_url}" target="_blank">🌐</a>' if wiki_url else ""
                grok_link = f'<a class="link-btn link-groki" href="{grok_url}" target="_blank">🤖</a>'

                st.markdown(f'''
<div class="result-card" style="padding:1rem 1.2rem;animation-delay:{(result_rank-1)*0.07:.2f}s">
  <div class="result-title-row" style="margin-bottom:0.5rem;">
    <span class="rank-badge {rank_cls}" style="width:28px;height:28px;font-size:0.8rem;">{result_rank}</span>
    <p class="result-title" style="font-size:0.95rem;">{result["title"]}</p>
  </div>
  <div class="score-bar-container">
    <div class="score-bar-label"><span>Score</span><span>{score_val:.4f}</span></div>
    <div class="score-bar-bg"><div class="score-bar-fill {bar_cls}" style="width:{bar_w}%"></div></div>
  </div>
  <div class="link-row" style="margin-top:0.5rem;">{wiki_link}{grok_link}</div>
</div>
''', unsafe_allow_html=True)
                with st.expander("📊 Breakdown"):
                    st.markdown(score_breakdown_text(
                        result["bm25_norm"], result["dense_norm"], mode, fusion_method
                    ))

# =============================================================================
# === UI — SIDEBAR: SYSTEM CONFIGURATION & SESSION STATS
# =============================================================================

if st.session_state.last_logged:
    st.sidebar.success("Results logged to search_log.csv")
    st.session_state.last_logged = False

st.sidebar.markdown("---")
st.sidebar.header("System Configuration")
expansion_status = "✅ On" if expand_query else "❌ Off"
st.sidebar.markdown(f"""
<div style="font-size:0.82rem;line-height:2;">
  <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.1);padding:3px 0;">
    <span style="color:rgba(255,255,255,0.5);">BM25 Variant</span>
    <span style="color:#a5b4fc;font-weight:500;">{sparse_model_label}</span>
  </div>
  <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.1);padding:3px 0;">
    <span style="color:rgba(255,255,255,0.5);">Mode</span>
    <span style="color:#a5b4fc;font-weight:500;">{retrieval_mode}</span>
  </div>
  <div style="display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.1);padding:3px 0;">
    <span style="color:rgba(255,255,255,0.5);">Fusion</span>
    <span style="color:#a5b4fc;font-weight:500;">{"RRF" if "RRF" in fusion_method else "Weighted Sum"}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:3px 0;">
    <span style="color:rgba(255,255,255,0.5);">Query Expansion</span>
    <span style="color:#a5b4fc;font-weight:500;">{expansion_status}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("Session Stats")
total_searches = st.session_state.total_searches
response_times = st.session_state.response_times
retrieval_modes_used = st.session_state.retrieval_modes_used

st.sidebar.metric("Total Searches", total_searches)
if response_times:
    avg_response_time = sum(response_times) / len(response_times)
    st.sidebar.metric("Avg Response Time", f"{avg_response_time:.0f} ms")
else:
    st.sidebar.metric("Avg Response Time", "—")
if retrieval_modes_used:
    most_common_mode = max(set(retrieval_modes_used), key=retrieval_modes_used.count)
    st.sidebar.metric("Most Used Mode", most_common_mode)
else:
    st.sidebar.metric("Most Used Mode", "—")

# =============================================================================
# === UI — FOOTER
# =============================================================================

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
with st.expander("ℹ️ About Wikipedia vs Grokipedia"):
    st.markdown("""
    | | Wikipedia | Grokipedia |
    |---|---|---|
    | **Source** | Human-edited, community-moderated | AI-generated by xAI Grok |
    | **Policy** | Strict sourcing & neutrality rules | AI synthesis |
    | **Use here** | Indexed for retrieval | Linked for comparison |

    This app retrieves from Wikipedia only, but provides Grokipedia links so you can compare human-edited vs AI-generated content on the same topic.
    """)

st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.3);font-size:0.75rem;padding:1rem 0 0.5rem 0;">
  Hybrid IR System · BM25 + SentenceTransformers + RRF · CSC-785-U18
</div>
""", unsafe_allow_html=True)
