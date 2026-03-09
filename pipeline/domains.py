"""
Steps 8.5–8.6 — Macro-domain mapping.

Reads fine-cluster assignments from SQLite (written by topics.py) and
meta-clusters the K fine-cluster centroids into M macro-domains using
KMeans on the mean reduced-space vectors.

Auto-labels each macro-domain by aggregating top TF-IDF terms across
its constituent fine clusters (weighted by cluster size).

Writes to SQLite:
    node_to_macro_domain  (node_id TEXT, macro_domain INT)

Writes to out_dir:
    macro_cluster_map.csv           — fine_cluster → macro_domain
    macro_domain_summary.csv        — macro domain sizes + top terms
    macro_monthly_metrics.csv       — monthly user entropy + JS divergence
    macro_monthly_domain_shares.csv — monthly user/assistant share per domain
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as shannon_entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_N_MACRO           = 8
TOP_TERMS_FINE            = 20
TOP_TERMS_MACRO           = 25
RANDOM_SEED               = 42
MIN_MSGS_PER_ROLE         = 50   # per-role minimum for JS divergence
MIN_USER_MSGS_PER_MONTH   = 50   # months below this are excluded from all monthly outputs


# ─────────────────────────────────────────────────────────────────────────────
# Label quality helpers
# ─────────────────────────────────────────────────────────────────────────────

# Conversational fillers that slip through sklearn's built-in "english" stop
# words.  Applied only at *label generation* time — the underlying topic model
# (TF-IDF fit, SVD, KMeans) is completely unaffected.
_LABEL_STOPWORDS: frozenset = frozenset({
    # Filler verbs
    "let", "get", "got", "use", "used", "make", "made", "go", "going",
    "know", "think", "want", "need", "see", "try", "help", "say", "said",
    "tell", "take", "give", "gave", "put", "set", "ask", "work", "look",
    "come", "came", "done", "doing", "feel", "felt", "keep", "kept",
    "run", "ran", "start", "stop", "move", "like", "liked", "yes", "sure",
    # Degree / hedging adverbs
    "just", "really", "actually", "basically", "literally", "definitely",
    "probably", "maybe", "quite", "rather", "pretty", "very", "simply",
    "mostly", "mainly", "generally", "usually", "often", "always", "never",
    # Conversational tokens
    "ok", "okay", "right", "great", "good", "nice", "fine", "cool",
    "thanks", "thank", "please", "hi", "hello", "hey", "bye", "sorry",
    # Contracted negations (apostrophe stripped by tokeniser)
    "don", "didn", "doesn", "isn", "wasn", "wouldn", "couldn",
    "shouldn", "haven", "hadn", "aren", "weren",
    # Vague common nouns
    "thing", "things", "something", "anything", "everything", "nothing",
    "way", "ways", "time", "times", "bit", "lot", "lots", "kind", "type",
    "point", "part", "parts", "place", "case", "cases",
    # Contraction fragments
    "ve", "ll", "re",
})


def _clean_label_terms(terms: list[str], n: int = 4) -> list[str]:
    """
    Return up to *n* label-quality terms from a ranked term list.

    Priority rules:
      1. Bigrams (contain a space) are always preferred — more semantically specific.
      2. Unigrams ≥ 4 chars that are not in _LABEL_STOPWORDS.
      3. Short unigrams (2–3 chars) kept only if all-uppercase (acronyms like AI, EU).

    Scans the full ranked list so that useful terms deep in the list are
    surfaced when the top terms are all fillers.
    """
    seen_roots: set[str] = set()
    result: list[str] = []

    for t in terms:
        if len(result) >= n:
            break

        t_lower = t.lower().strip()

        # Always keep bigrams
        if " " in t:
            result.append(t)
            seen_roots.add(t_lower)
            continue

        # Skip filler stop words
        if t_lower in _LABEL_STOPWORDS:
            continue

        # Skip very short tokens unless they are acronyms (all-caps, len ≥ 2)
        if len(t) <= 2 and not (t.isupper() and len(t) == 2):
            continue

        # Soft deduplication: skip if this term is a prefix/suffix of one already chosen
        # (e.g. "chapter" if "chapters" already selected)
        if any(t_lower.startswith(r) or r.startswith(t_lower) for r in seen_roots):
            continue

        result.append(t)
        seen_roots.add(t_lower)

    return result


def run(
    db_path: str | Path,
    out_dir: str | Path,
    n_macro: int = DEFAULT_N_MACRO,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Build macro-domain hierarchy from fine cluster assignments in SQLite.

    Requires topics.run() to have been called first.

    Args:
        db_path:     SQLite database
        out_dir:     Directory for output CSVs
        n_macro:     Number of macro-domains (default 8)
        progress_cb: Optional callable(fraction 0–1, status_string)

    Returns:
        Summary dict: n_macro, domain labels, monthly metrics shape
    """
    def _cb(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path)

    # ── 1. Load messages + fine cluster assignments ───────────────────────────
    _cb(0.03, "Loading messages and cluster assignments…")
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT m.node_id, m.role, m.year_month, m.text,
                  n.cluster_id AS fine_cluster
           FROM messages m
           JOIN node_to_fine_cluster n ON m.node_id = n.node_id
           WHERE m.role IN ('user', 'assistant')
             AND m.text IS NOT NULL AND LENGTH(m.text) > 0""",
        con,
    )
    con.close()

    if df.empty:
        raise ValueError(
            "No cluster data found. Run topics.run() before domains.run()."
        )

    k_fine = int(df["fine_cluster"].max()) + 1
    texts = df["text"].astype(str).tolist()

    # ── 2. Re-fit TF-IDF to get term vocabulary for labelling ─────────────────
    _cb(0.10, "Vectorising for domain labelling…")
    vec = TfidfVectorizer(
        max_features=60_000,
        min_df=3,
        max_df=0.6,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    # ── 3. SVD reduce ─────────────────────────────────────────────────────────
    n_components = min(200, X.shape[1] - 1)
    _cb(0.20, f"Reducing dimensions (SVD → {n_components})…")
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    X_red = normalize(svd.fit_transform(X))

    fine_labels = df["fine_cluster"].values

    # ── 4. Compute fine-cluster centroids + top terms ──────────────────────────
    _cb(0.35, "Computing fine-cluster centroids…")
    centroids = np.zeros((k_fine, X_red.shape[1]), dtype=float)
    fine_top_terms: dict[int, list[str]] = {}

    for c in range(k_fine):
        idx = np.where(fine_labels == c)[0]
        if len(idx) == 0:
            fine_top_terms[c] = []
            continue
        centroids[c] = X_red[idx].mean(axis=0)
        mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[-TOP_TERMS_FINE:][::-1]
        fine_top_terms[c] = terms[top_idx].tolist()

    # ── 5. Meta-cluster fine centroids → macro-domains ────────────────────────
    m = min(n_macro, k_fine)
    _cb(0.50, f"Meta-clustering {k_fine} fine clusters → {m} macro-domains…")
    km_macro = KMeans(n_clusters=m, random_state=RANDOM_SEED, n_init=20)
    macro_labels = km_macro.fit_predict(centroids)

    fine_to_macro: dict[int, int] = {c: int(macro_labels[c]) for c in range(k_fine)}
    df["macro_domain"] = df["fine_cluster"].map(fine_to_macro)

    # ── 6. Auto-label macro-domains ───────────────────────────────────────────
    _cb(0.60, "Labelling macro-domains…")
    macro_rows = []
    for md in range(m):
        fine_in_md = [c for c in range(k_fine) if fine_to_macro[c] == md]
        term_scores: dict[str, float] = {}
        for c in fine_in_md:
            c_size = int((df["fine_cluster"] == c).sum())
            for rank, t in enumerate(fine_top_terms.get(c, [])):
                w = c_size * (TOP_TERMS_FINE - rank)
                term_scores[t] = term_scores.get(t, 0) + w

        top_terms = [
            t for t, _ in sorted(
                term_scores.items(), key=lambda x: x[1], reverse=True
            )[:TOP_TERMS_MACRO]
        ]
        clean      = _clean_label_terms(top_terms, n=4)
        auto_label = ", ".join(clean) if clean else f"domain_{md}"

        macro_rows.append({
            "macro_domain":     md,
            "size":             int((df["macro_domain"] == md).sum()),
            "fine_clusters":    ",".join(map(str, fine_in_md)),
            "auto_label":       auto_label,
            "top_terms":        ", ".join(top_terms),
        })

    macro_summary = pd.DataFrame(macro_rows).sort_values("size", ascending=False)

    # ── 7. Write node_to_macro_domain to SQLite ───────────────────────────────
    _cb(0.68, "Writing macro-domain assignments to database…")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
        DROP TABLE IF EXISTS node_to_macro_domain;
        CREATE TABLE node_to_macro_domain (
            node_id      TEXT PRIMARY KEY,
            macro_domain INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_ntmd_domain
            ON node_to_macro_domain(macro_domain);
    """)
    cur.executemany(
        "INSERT INTO node_to_macro_domain (node_id, macro_domain) VALUES (?, ?)",
        zip(df["node_id"].tolist(), df["macro_domain"].tolist()),
    )
    con.commit()
    con.close()

    # ── 8. Monthly macro metrics ──────────────────────────────────────────────
    _cb(0.78, "Computing monthly macro metrics…")
    # Only include months that have enough user messages to be meaningful
    all_months     = sorted(df["year_month"].dropna().unique())
    user_per_month = df[df["role"] == "user"].groupby("year_month").size()
    months         = [
        mo for mo in all_months
        if user_per_month.get(mo, 0) >= MIN_USER_MSGS_PER_MONTH
    ]
    metrics_rows = []
    shares_rows  = []

    for month in months:
        g  = df[df["year_month"] == month]
        ug = g[g["role"] == "user"]
        ag = g[g["role"] == "assistant"]

        u = np.array([
            (ug["macro_domain"] == d).sum() for d in range(m)
        ], dtype=float)
        a = np.array([
            (ag["macro_domain"] == d).sum() for d in range(m)
        ], dtype=float)

        u_share = u / u.sum() if u.sum() > 0 else np.zeros(m)
        a_share = a / a.sum() if a.sum() > 0 else np.zeros(m)

        H  = float(shannon_entropy(u_share)) if u.sum() > 0 else np.nan
        js = float(jensenshannon(u_share, a_share)) \
            if (u.sum() >= MIN_MSGS_PER_ROLE and a.sum() >= MIN_MSGS_PER_ROLE) \
            else np.nan

        metrics_rows.append({
            "year_month":          month,
            "user_msgs":           int(len(ug)),
            "asst_msgs":           int(len(ag)),
            "macro_entropy_user":  H,
            "macro_js_divergence": js,
        })

        for d in range(m):
            shares_rows.append({
                "year_month":      month,
                "macro_domain":    d,
                "user_share":      float(u_share[d]),
                "asst_share":      float(a_share[d]),
            })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("year_month")
    shares_df  = pd.DataFrame(shares_rows).sort_values(["year_month", "macro_domain"])

    # ── 9. Write CSVs ─────────────────────────────────────────────────────────
    _cb(0.92, "Writing CSVs…")
    map_df = pd.DataFrame([
        {"fine_cluster": c, "macro_domain": fine_to_macro[c],
         "fine_top_terms": ", ".join(fine_top_terms.get(c, []))}
        for c in range(k_fine)
    ]).sort_values(["macro_domain", "fine_cluster"])

    macro_summary.to_csv(out_dir / "macro_domain_summary.csv",         index=False)
    map_df.to_csv(       out_dir / "macro_cluster_map.csv",             index=False)
    metrics_df.to_csv(   out_dir / "macro_monthly_metrics.csv",         index=False)
    shares_df.to_csv(    out_dir / "macro_monthly_domain_shares.csv",   index=False)

    _cb(1.0, "Macro-domain mapping complete.")

    return {
        "n_macro":         m,
        "domain_labels":   macro_summary["auto_label"].tolist(),
        "months_computed": len(metrics_df),
    }
