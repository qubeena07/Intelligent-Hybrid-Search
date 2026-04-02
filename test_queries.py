"""
test_queries.py
===============
Evaluation query set for the Hybrid IR System (CSC-785-U18).

Contains 50 Wikipedia-appropriate search queries across four categories:
  - factual    (15): specific facts, dates, people, places
  - conceptual (15): how/why questions about science, history, technology
  - ambiguous  (10): short 1–2 word queries with multiple valid interpretations
  - technical  (10): domain-specific multi-word phrases

Usage
-----
    from test_queries import TEST_QUERIES, RELEVANCE_TEMPLATE

After running the IR system against these queries, populate each entry in
RELEVANCE_TEMPLATE with the titles of documents judged relevant by a human
assessor, then use the filled template to compute Precision@K, Recall, and MAP.
"""

# =============================================================================
# === QUERY SET
# =============================================================================

TEST_QUERIES = [

    # -------------------------------------------------------------------------
    # FACTUAL (15) — specific facts, dates, named entities, places
    # -------------------------------------------------------------------------
    {"query": "When was the Eiffel Tower constructed and by whom",              "type": "factual"},
    {"query": "What country did Marie Curie emigrate from to study in Paris",   "type": "factual"},
    {"query": "Year the Berlin Wall fell",                                       "type": "factual"},
    {"query": "Capital city of Australia",                                       "type": "factual"},
    {"query": "First human landing on the Moon mission name",                   "type": "factual"},
    {"query": "What language is spoken in Brazil",                              "type": "factual"},
    {"query": "Height of Mount Everest in metres",                              "type": "factual"},
    {"query": "Who invented the telephone Alexander Graham Bell",               "type": "factual"},
    {"query": "Treaty that ended World War One",                                "type": "factual"},
    {"query": "Population of Tokyo metropolitan area",                          "type": "factual"},
    {"query": "Year Charles Darwin published On the Origin of Species",         "type": "factual"},
    {"query": "Which ocean is the deepest Mariana Trench location",             "type": "factual"},
    {"query": "Founder of the Microsoft Corporation",                           "type": "factual"},
    {"query": "What element has atomic number 79",                              "type": "factual"},
    {"query": "Year the Soviet Union dissolved",                                "type": "factual"},

    # -------------------------------------------------------------------------
    # CONCEPTUAL (15) — how/why questions, mechanisms, causes
    # -------------------------------------------------------------------------
    {"query": "How does the human immune system recognise pathogens",           "type": "conceptual"},
    {"query": "Why did the Roman Empire decline and fall",                      "type": "conceptual"},
    {"query": "How does CRISPR gene editing technology work",                   "type": "conceptual"},
    {"query": "What causes the northern lights aurora borealis",                "type": "conceptual"},
    {"query": "How did the Industrial Revolution change labour conditions",      "type": "conceptual"},
    {"query": "Why is the sky blue Rayleigh scattering explanation",            "type": "conceptual"},
    {"query": "How do vaccines produce immunity in the human body",             "type": "conceptual"},
    {"query": "What caused the 2008 global financial crisis",                   "type": "conceptual"},
    {"query": "How does natural selection drive speciation",                    "type": "conceptual"},
    {"query": "Why do tectonic plates move continental drift mechanism",        "type": "conceptual"},
    {"query": "How does the internet route data packets",                       "type": "conceptual"},
    {"query": "What is the role of mitochondria in cellular respiration",       "type": "conceptual"},
    {"query": "How did the printing press change the spread of information",    "type": "conceptual"},
    {"query": "Why does the Moon cause ocean tides gravitational pull",         "type": "conceptual"},
    {"query": "How do black holes form from collapsing stars",                  "type": "conceptual"},

    # -------------------------------------------------------------------------
    # AMBIGUOUS (10) — 1–2 word queries with multiple valid interpretations
    # -------------------------------------------------------------------------
    {"query": "bank",       "type": "ambiguous"},  # financial institution vs. river bank
    {"query": "python",     "type": "ambiguous"},  # programming language vs. snake species
    {"query": "mercury",    "type": "ambiguous"},  # planet vs. element vs. Roman god
    {"query": "crane",      "type": "ambiguous"},  # construction machine vs. bird species
    {"query": "java",       "type": "ambiguous"},  # programming language vs. Indonesian island vs. coffee
    {"query": "plate",      "type": "ambiguous"},  # tectonic plate vs. dining plate vs. metal plate
    {"query": "falcon",     "type": "ambiguous"},  # bird of prey vs. SpaceX rocket vs. Ford car
    {"query": "cell",       "type": "ambiguous"},  # biological cell vs. prison cell vs. battery cell
    {"query": "Amazon",     "type": "ambiguous"},  # river vs. e-commerce company vs. rainforest
    {"query": "corona",     "type": "ambiguous"},  # solar corona vs. beer brand vs. virus

    # -------------------------------------------------------------------------
    # TECHNICAL (10) — domain-specific multi-word phrases
    # -------------------------------------------------------------------------
    {"query": "Byzantine fault tolerance distributed systems",                  "type": "technical"},
    {"query": "Krebs cycle mechanism citric acid intermediates",                "type": "technical"},
    {"query": "transformer self-attention mechanism natural language processing","type": "technical"},
    {"query": "Hamming distance error correcting codes binary",                 "type": "technical"},
    {"query": "mRNA splicing spliceosome intron exon processing",               "type": "technical"},
    {"query": "Fourier transform signal decomposition frequency domain",        "type": "technical"},
    {"query": "PageRank algorithm link analysis web graph",                     "type": "technical"},
    {"query": "Nash equilibrium game theory mixed strategy",                    "type": "technical"},
    {"query": "renormalization group quantum field theory divergence",          "type": "technical"},
    {"query": "apoptosis caspase cascade programmed cell death pathway",        "type": "technical"},
]

# =============================================================================
# === RELEVANCE TEMPLATE
# =============================================================================
# After running the IR system, a human assessor should fill each "relevant_titles"
# list with the Wikipedia article titles judged relevant to that query.
# These judgements are then used to compute Precision@K, Recall@K, and MAP.

RELEVANCE_TEMPLATE = [
    {"query": entry["query"], "type": entry["type"], "relevant_titles": []}
    for entry in TEST_QUERIES
]

# =============================================================================
# === QUICK SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    from collections import Counter

    type_counts = Counter(entry["type"] for entry in TEST_QUERIES)
    print(f"Total queries : {len(TEST_QUERIES)}")
    for query_type, count in sorted(type_counts.items()):
        print(f"  {query_type:<12}: {count}")

    assert len(TEST_QUERIES) == 50, "Expected exactly 50 queries"
    assert len(RELEVANCE_TEMPLATE) == 50, "RELEVANCE_TEMPLATE length mismatch"
    assert type_counts["factual"]    == 15
    assert type_counts["conceptual"] == 15
    assert type_counts["ambiguous"]  == 10
    assert type_counts["technical"]  == 10
    print("\nAll assertions passed.")
