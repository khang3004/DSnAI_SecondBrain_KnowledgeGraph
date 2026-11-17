#!/usr/bin/env python3
"""Library PDF indexer, deduplicator, TOC extractor, and inverted index builder.

Scans all PDF textbooks and monographs inside ``./books_monographs/`` using
PyMuPDF (``pymupdf``), detects exact and near-duplicate documents, extracts
embedded or inferred Tables of Contents, computes page-level keyword and
topic-axis embeddings, builds a cross-book inverted index, and saves the
structured result to ``./data/book_index.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata

import pymupdf


# Common English + academic + Vietnamese stopwords excluded from keyword indexing.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "also", "am",
        "an", "and", "any", "are", "aren", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can",
        "cannot", "could", "did", "do", "does", "doing", "don", "down", "during",
        "each", "either", "else", "etc", "even", "every", "few", "for", "from",
        "further", "get", "give", "given", "had", "has", "have", "having", "he",
        "hence", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "however", "if", "in", "indeed", "into", "is", "it", "its", "itself",
        "just", "let", "like", "made", "make", "many", "may", "me", "might",
        "more", "most", "much", "must", "my", "myself", "neither", "new", "no",
        "nor", "not", "now", "of", "off", "on", "once", "one", "only", "or",
        "other", "our", "ours", "ourselves", "out", "over", "own", "part", "per",
        "put", "rather", "same", "see", "seem", "seen", "several", "she", "should",
        "show", "shown", "shows", "since", "so", "some", "still", "such", "take",
        "taken", "than", "that", "the", "their", "theirs", "them", "themselves",
        "then", "there", "these", "they", "this", "those", "though", "three",
        "through", "thus", "to", "too", "two", "under", "until", "up", "upon",
        "us", "use", "used", "using", "various", "very", "via", "was", "we",
        "well", "were", "what", "when", "where", "whereas", "whether", "which",
        "while", "who", "whom", "whose", "why", "will", "with", "within",
        "without", "would", "you", "your", "yours", "yourself",
        # Publishing / structural boilerplate
        "chapter", "section", "subsection", "figure", "fig", "table", "page",
        "pages", "edition", "press", "university", "copyright", "rights",
        "reserved", "published", "publisher", "isbn", "issn", "doi", "arxiv",
        "preprint", "printed", "contents", "index", "preface", "acknowledgments",
        "references", "bibliography", "appendix", "exercise", "exercises",
        "problem", "problems", "solution", "solutions", "example", "examples",
        "note", "notes", "remark", "remarks", "corollary", "proof", "definition",
        "lemma", "proposition", "eq", "equation", "equations", "http", "https",
        "www", "com", "org", "edu", "pdf", "vol", "pp", "et", "al",
        # Vietnamese common stopwords (for Tiep Vu's ML book)
        "va", "cua", "la", "cac", "trong", "cho", "mot", "nhung", "duoc", "khi",
        "voi", "tai", "den", "tu", "nhu", "nay", "do", "neu", "thi", "ma", "ta",
        "chung", "ban", "phan", "bai", "chuong", "hinh", "cong", "thuc", "vi",
        "du", "khong", "co", "the", "se", "da", "rat", "nhieu", "hon", "nhat",
    }
)

# Stopwords trimmed only when constructing concise book slugs from filenames.
SLUG_FILLER_TOKENS: frozenset[str] = frozenset(
    {"a", "an", "the", "of", "for", "to", "and", "in", "on", "by", "with", "edition", "second", "2nd"}
)

# Canonical topic axes used to compute compact dense page-level topic embeddings.
TOPIC_AXES: dict[str, tuple[str, ...]] = {
    "linear_algebra": (
        "matrix", "matrices", "vector", "eigenvalue", "eigenvector", "svd",
        "singular value", "orthogonal", "transpose", "determinant", "rank",
        "subspace", "inner product", "norm", "positive definite", "spectral",
        "projection", "qr", "cholesky", "lu decomposition",
    ),
    "calculus_optimization": (
        "gradient", "gradient descent", "stochastic gradient", "hessian",
        "jacobian", "convex", "convexity", "lagrange", "lagrangian", "duality",
        "kkt", "adam", "momentum", "nesterov", "learning rate", "line search",
        "convergence", "lipschitz", "subgradient", "newton", "quasi newton",
        "bfgs", "lbfgs", "saddle point", "unconstrained", "constrained",
    ),
    "probability_statistics": (
        "probability", "random variable", "expectation", "variance", "covariance",
        "gaussian", "normal distribution", "bernoulli", "poisson", "bayes",
        "bayesian", "posterior", "prior", "likelihood", "mle", "map",
        "markov chain", "monte carlo", "mcmc", "gibbs", "metropolis",
        "central limit", "law of large", "hypothesis", "confidence interval",
    ),
    "information_theory_ot": (
        "entropy", "cross entropy", "kl divergence", "kullback leibler",
        "mutual information", "jensen shannon", "f divergence", "wasserstein",
        "optimal transport", "kantorovich", "monge", "sinkhorn", "earth mover",
        "bregman", "information", "channel capacity",
    ),
    "classical_ml": (
        "linear regression", "logistic regression", "ridge", "lasso", "svm",
        "support vector", "kernel", "decision tree", "random forest", "boosting",
        "xgboost", "adaboost", "kmeans", "clustering", "pca", "gmm",
        "gaussian mixture", "em algorithm", "expectation maximization",
        "naive bayes", "knn", "nearest neighbor", "cross validation", "bias variance",
    ),
    "deep_learning_core": (
        "neural network", "mlp", "perceptron", "backpropagation", "activation",
        "relu", "sigmoid", "softmax", "batch normalization", "layer normalization",
        "dropout", "residual", "resnet", "cnn", "convolutional", "pooling",
        "rnn", "lstm", "gru", "autoencoder", "initialization", "xavier",
        "kan", "kolmogorov arnold", "spline", "universal approximation",
    ),
    "generative_models": (
        "vae", "variational autoencoder", "elbo", "variational inference", "gan",
        "generative adversarial", "normalizing flow", "diffusion", "ddpm",
        "score matching", "sde", "langevin", "fokker planck", "classifier free",
        "latent diffusion", "denoising", "ebm", "energy based",
    ),
    "transformers_llms": (
        "transformer", "attention", "self attention", "multi head", "query key",
        "positional encoding", "rope", "tokenization", "bpe", "language model",
        "llm", "autoregressive", "gpt", "bert", "pretraining", "fine tuning",
        "lora", "peft", "kv cache", "flash attention", "mamba", "state space",
        "ssm", "selective scan", "context length", "perplexity", "beam search",
    ),
    "reinforcement_learning": (
        "reinforcement learning", "mdp", "markov decision", "bellman",
        "value function", "q learning", "sarsa", "td learning", "temporal difference",
        "monte carlo", "policy gradient", "reinforce", "actor critic", "ppo",
        "trpo", "dqn", "bandit", "ucb", "thompson sampling", "exploration",
        "rlhf", "dpo", "reward model", "preference", "bradley terry",
    ),
    "agentic_systems_mlops": (
        "agent", "agentic", "tool use", "function calling", "mcp",
        "model context protocol", "react", "chain of thought", "planning",
        "memory", "rag", "retrieval", "context engineering", "multi agent",
        "evaluation", "guardrail", "observability", "workflow", "n8n",
        "deployment", "monitoring", "feature store", "data drift", "mlops",
    ),
}

# Curated multi-word and single-word technical terms always preserved in the inverted index.
CURATED_TERMS: frozenset[str] = frozenset(
    term
    for terms in TOPIC_AXES.values()
    for term in terms
) | frozenset(
    {
        "backpropagation", "gradient descent", "stochastic gradient descent",
        "adamw", "rmsprop", "adagrad", "conjugate gradient", "newton raphson",
        "lagrange multiplier", "kkt conditions", "convex optimization",
        "singular value decomposition", "eigendecomposition", "spectral theorem",
        "principal component analysis", "positive semidefinite", "condition number",
        "bayes theorem", "maximum likelihood", "maximum a posteriori",
        "variational inference", "expectation maximization", "markov chain",
        "monte carlo", "importance sampling", "rejection sampling",
        "gibbs sampling", "metropolis hastings", "hamiltonian monte carlo",
        "kl divergence", "cross entropy", "mutual information", "wasserstein distance",
        "optimal transport", "sinkhorn algorithm", "kantorovich duality",
        "support vector machine", "kernel trick", "gaussian process",
        "hidden markov model", "kalman filter", "graphical model", "belief propagation",
        "convolutional neural network", "recurrent neural network", "residual network",
        "layer norm", "batch norm", "self attention", "multi head attention",
        "scaled dot product", "rotary position embedding", "kv cache",
        "state space model", "structured state space", "selective state space",
        "diffusion model", "score matching", "denoising diffusion",
        "flow matching", "normalizing flow", "variational autoencoder",
        "generative adversarial network", "kolmogorov arnold network",
        "markov decision process", "bellman equation", "dynamic programming",
        "temporal difference", "q learning", "policy gradient",
        "proximal policy optimization", "actor critic", "advantage function",
        "generalized advantage estimation", "multi armed bandit",
        "rlhf", "direct preference optimization", "reward modeling",
        "retrieval augmented generation", "model context protocol",
        "chain of thought", "react agent", "tool calling", "context engineering",
        "long term memory", "short term memory", "agent evaluation",
        "mlops", "feature engineering", "concept drift", "data leakage",
    }
)

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}")
PRINTED_TOC_HEADER_RE = re.compile(
    r"\b(table\s+of\s+contents|contents|m[uụ]c\s+l[uụ]c|talk\s+outline)\b", re.IGNORECASE
)
PRINTED_TOC_LINE_RE = re.compile(
    r"^(?P<title>(?:Chapter\s+\d+[:.\s]+|\d+(?:\.\d+){0,3}\s+|[A-Z]\.\d+\s+)?[^\n]{3,95}?)"
    r"[\s.·•…_-]{2,}(?P<page>\d{1,4})\s*$"
)
HEADING_NUMBERED_RE = re.compile(
    r"^(?:(?:Chapter|Part|Section|Appendix)\s+\d+(?:\.\d+)*[:.\s]+|"
    r"\d+(?:\.\d+){0,2}\s+)[A-ZÀ-Ỹ][^\n]{2,80}$"
)


@dataclass(slots=True)
class TocEntry:
    """Represents a single normalized Table of Contents entry."""

    level: int
    title: str
    page: int
    end_page: int


@dataclass(slots=True)
class DuplicateMatch:
    """Describes a duplicate or near-duplicate relationship between two books."""

    other_slug: str
    other_path: str
    match_type: str  # "exact_sha256" | "near_duplicate"
    title_similarity: float
    text_similarity: float
    page_count_diff: int


@dataclass(slots=True)
class PageRecord:
    """Compact page-level representation for indexing and concept mapping."""

    page: int
    label: str
    char_count: int
    image_count: int
    top_terms: list[str]
    topic_embedding: dict[str, float]


@dataclass(slots=True)
class BookRecord:
    """Complete structured index record for one PDF book/monograph."""

    book_slug: str
    file_path: str
    file_name: str
    library_category: str
    sha256: str
    file_size_bytes: int
    page_count: int
    is_duplicate: bool = False
    duplicate_of: str | None = None
    similarity_matches: list[DuplicateMatch] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    inferred_title: str = ""
    printed_page_offset: int = 0
    toc_source: str = "none"
    toc: list[TocEntry] = field(default_factory=list)
    top_keywords: list[dict[str, Any]] = field(default_factory=list)
    domain_profile: dict[str, float] = field(default_factory=dict)
    pages: list[PageRecord] = field(default_factory=list)


def fold_to_ascii(text: str) -> str:
    """Transliterate Unicode text (including Vietnamese) to lowercase ASCII."""
    # Replace Vietnamese đ/Đ before NFKD decomposition because it does not decompose.
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFKD", text)
    ascii_bytes = normalized.encode("ascii", "ignore")
    return ascii_bytes.decode("ascii").lower()


def clean_whitespace(text: str) -> str:
    """Collapse arbitrary whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def make_slug(file_path: Path, existing_slugs: set[str]) -> str:
    """Generate a concise, deterministic, collision-free ASCII book slug."""
    stem = fold_to_ascii(file_path.stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    raw_tokens = [tok for tok in stem.split("_") if tok]
    filtered_tokens = [tok for tok in raw_tokens if tok not in SLUG_FILLER_TOKENS]
    tokens = filtered_tokens if len(filtered_tokens) >= 2 else raw_tokens
    if len(tokens) > 6:
        tokens = tokens[:6]
    base_slug = "_".join(tokens) or "book"
    if len(base_slug) > 48:
        base_slug = base_slug[:48].rstrip("_")

    candidate = base_slug
    counter = 2
    while candidate in existing_slugs:
        suffix = f"_{counter}"
        candidate = f"{base_slug[:48 - len(suffix)]}{suffix}"
        counter += 1
    existing_slugs.add(candidate)
    return candidate


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest of a file using streaming reads."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_title_for_matching(title: str) -> str:
    """Normalize a book title or filename stem for fuzzy similarity checks."""
    folded = fold_to_ascii(title)
    folded = re.sub(
        r"\b(second|2nd|third|3rd|first|1st|edition|20\d{2}|pdf|v\d+)\b",
        " ",
        folded,
    )
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return clean_whitespace(folded)


def extract_tokens_and_bigrams(text: str) -> tuple[list[str], list[str]]:
    """Extract normalized unigrams and informative bigrams from page text."""
    folded = fold_to_ascii(text)
    raw_tokens = TOKEN_RE.findall(folded)
    unigrams: list[str] = []
    for tok in raw_tokens:
        tok_clean = tok.strip("_-")
        if len(tok_clean) < 2 or tok_clean.isdigit() or tok_clean in STOPWORDS:
            continue
        if len(tok_clean) > 32:
            continue
        unigrams.append(tok_clean)

    bigrams: list[str] = [
        f"{unigrams[idx]} {unigrams[idx + 1]}"
        for idx in range(len(unigrams) - 1)
    ]
    return unigrams, bigrams


def compute_topic_embedding(
    unigram_counts: Counter[str],
    bigram_counts: Counter[str],
) -> dict[str, float]:
    """Project page term counts onto normalized canonical AI/ML topic axes."""
    scores: dict[str, float] = {}
    for axis, keywords in TOPIC_AXES.items():
        axis_score = 0.0
        for kw in keywords:
            if " " in kw:
                count = bigram_counts.get(kw, 0)
                if count > 0:
                    # Multi-word phrases carry higher specificity than single words.
                    axis_score += 2.0 * (1.0 + math.log1p(count))
            else:
                count = unigram_counts.get(kw, 0)
                if count > 0:
                    axis_score += 1.0 + math.log1p(count)
        if axis_score > 0.0:
            scores[axis] = axis_score

    norm = math.sqrt(sum(val * val for val in scores.values()))
    if norm == 0.0:
        return {}
    return {
        axis: round(val / norm, 4)
        for axis, val in sorted(scores.items(), key=lambda item: -item[1])
        if val / norm >= 0.05
    }


def assign_toc_end_pages(
    entries: list[tuple[int, str, int]],
    page_count: int,
) -> list[TocEntry]:
    """Compute ``end_page`` boundaries for hierarchical TOC items."""
    valid: list[tuple[int, str, int]] = []
    for level, title, page_num in entries:
        cleaned = clean_whitespace(title)
        if not cleaned:
            continue
        clamped_page = max(1, min(page_count, page_num))
        valid.append((max(1, level), cleaned, clamped_page))

    result: list[TocEntry] = []
    total = len(valid)
    for idx, (level, title, start_page) in enumerate(valid):
        end_page = page_count
        for next_idx in range(idx + 1, total):
            next_level, _, next_page = valid[next_idx]
            if next_level <= level:
                end_page = next_page - 1 if next_page > start_page else start_page
                break
        end_page = max(start_page, min(page_count, end_page))
        result.append(
            TocEntry(
                level=level,
                title=title,
                page=start_page,
                end_page=end_page,
            )
        )
    return result


def infer_printed_toc(
    page_texts: list[str],
    max_scan_pages: int = 25,
) -> tuple[list[TocEntry], int]:
    """Extract TOC from printed Contents pages, handling dotted and two-line forms."""
    page_count = len(page_texts)
    raw_items: list[tuple[int, str, int]] = []
    in_contents = False
    toc_pages_seen = 0
    last_toc_idx = 0

    for idx in range(min(max_scan_pages, page_count)):
        text = page_texts[idx]
        if not in_contents and PRINTED_TOC_HEADER_RE.search(text[:800]):
            in_contents = True
        if not in_contents:
            continue

        toc_pages_seen += 1
        if toc_pages_seen > 12:
            break

        # Normalize non-breaking spaces used in some textbooks (e.g., Goodfellow).
        lines = [clean_whitespace(l.replace("\xa0", " ")) for l in text.splitlines() if l.strip()]
        matched_on_page = 0
        line_idx = 0
        while line_idx < len(lines):
            line = lines[line_idx]
            match = PRINTED_TOC_LINE_RE.match(line)
            if match:
                title = clean_whitespace(match.group("title")).rstrip(".·•…_- ")
                printed_page = int(match.group("page"))
                line_idx += 1
            elif (
                line_idx + 1 < len(lines)
                and lines[line_idx + 1].isdigit()
                and 3 <= len(line) <= 90
                and not line.isdigit()
                and not line.endswith((".", ",", ";"))
                and not PRINTED_TOC_HEADER_RE.search(line)
                and int(lines[line_idx + 1]) >= (raw_items[-1][2] if raw_items else 1)
            ):
                # Handles modern brochures/guides with title on one line and page number below.
                title = line
                printed_page = int(lines[line_idx + 1])
                line_idx += 2
            elif (
                line_idx + 1 < len(lines)
                and re.match(r"^\d+(?:\.\d+){0,3}$", line)
                and not lines[line_idx + 1].isdigit()
            ):
                # Handles Goodfellow-style split lines: "1.1" \n "Who Should Read... . . 8"
                combined = f"{line} {lines[line_idx + 1]}"
                comb_match = PRINTED_TOC_LINE_RE.match(combined)
                if comb_match:
                    title = clean_whitespace(comb_match.group("title")).rstrip(".·•…_- ")
                    printed_page = int(comb_match.group("page"))
                    line_idx += 2
                else:
                    line_idx += 1
                    continue
            else:
                line_idx += 1
                continue

            if printed_page < 1 or printed_page > page_count + 50:
                continue
            num_match = re.match(r"^(\d+(?:\.\d+)*)\b", title)
            if num_match:
                level = min(3, num_match.group(1).count(".") + 1)
            elif re.match(r"^(?:Chapter|Part)\b", title, re.IGNORECASE):
                level = 1
            else:
                level = 2
            raw_items.append((level, title, printed_page))
            matched_on_page += 1
            last_toc_idx = idx

        if toc_pages_seen > 1 and matched_on_page == 0 and len(raw_items) >= 5:
            break

    if len(raw_items) < 3:
        return [], 0

    # Start body search strictly after the TOC pages so TOC lines cannot self-match.
    body_start_idx = min(page_count - 1, last_toc_idx + 1)
    offsets: list[int] = []
    for _, title, printed_page in raw_items[:15]:
        core_title = re.sub(r"^(?:Chapter\s+\d+[:.\s]+|\d+(?:\.\d+)*\s+)", "", title)
        core_norm = fold_to_ascii(core_title)
        if len(core_norm) < 6:
            continue
        search_start = max(body_start_idx, printed_page - 3)
        search_end = min(page_count, printed_page + 45)
        for pdf_idx in range(search_start, search_end):
            snippet = fold_to_ascii(page_texts[pdf_idx][:700])
            if core_norm in snippet:
                offsets.append((pdf_idx + 1) - printed_page)
                break

    page_offset = Counter(offsets).most_common(1)[0][0] if offsets else 0
    shifted = [
        (level, title, printed_page + page_offset)
        for level, title, printed_page in raw_items
        if 1 <= printed_page + page_offset <= page_count
    ]
    if len(shifted) < 3:
        return [], 0
    return assign_toc_end_pages(shifted, page_count), page_offset


def infer_layout_headings(doc: pymupdf.Document, page_texts: list[str]) -> list[TocEntry]:
    """Infer section or slide titles from font sizes, position, and numbered headings."""
    page_count = len(page_texts)
    char_counts = sorted(len(t) for t in page_texts)
    median_chars = char_counts[len(char_counts) // 2] if char_counts else 0
    is_slide_deck = median_chars < 950

    font_sizes: list[float] = []
    sample_indices = list(range(min(page_count, 25)))
    if page_count > 25:
        step = max(1, page_count // 25)
        sample_indices = list(range(0, page_count, step))[:25]

    for p_idx in sample_indices:
        page_dict = doc[p_idx].get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if len(text) >= 12:
                        font_sizes.append(float(span.get("size", 10.0)))

    body_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0
    heading_threshold = body_size * (1.12 if is_slide_deck else 1.18)

    candidates: list[tuple[int, float, str]] = []
    for p_idx in range(page_count):
        page = doc[p_idx]
        height = float(page.rect.height) or 792.0
        page_dict = page.get_text("dict")
        slide_top_captured = False

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                bbox = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
                top_cutoff = 0.015 if is_slide_deck else 0.07
                bottom_cutoff = 0.94
                if bbox[1] < top_cutoff * height or bbox[3] > bottom_cutoff * height:
                    continue
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = clean_whitespace("".join(s.get("text", "") for s in spans))
                if len(line_text) < 4 or len(line_text) > 95 or line_text.isdigit():
                    continue
                max_size = max(float(s.get("size", 0.0)) for s in spans)
                is_bold = any(bool(s.get("flags", 0) & 16) for s in spans)
                is_numbered = bool(HEADING_NUMBERED_RE.match(line_text))

                if is_slide_deck:
                    if not slide_top_captured and bbox[1] <= 0.18 * height and (max_size >= body_size * 1.05 or is_bold):
                        candidates.append((p_idx + 1, max_size, line_text))
                        slide_top_captured = True
                    continue

                if max_size >= heading_threshold or (
                    is_numbered and (is_bold or max_size >= body_size * 1.05)
                ):
                    candidates.append((p_idx + 1, max_size, line_text))

    if not candidates:
        return []

    deduped: list[tuple[int, float, str]] = []
    for p_num, size, text in candidates:
        if (
            deduped
            and deduped[-1][0] == p_num
            and abs(deduped[-1][1] - size) < 0.8
            and len(deduped[-1][2]) + len(text) < 110
        ):
            prev_p, prev_s, prev_t = deduped[-1]
            deduped[-1] = (prev_p, prev_s, f"{prev_t} {text}")
        else:
            deduped.append((p_num, size, text))

    # For slide decks, collapse consecutive slides that share the exact same header.
    if is_slide_deck:
        collapsed: list[tuple[int, float, str]] = []
        for p_num, size, text in deduped:
            if collapsed and collapsed[-1][2].lower() == text.lower():
                continue
            collapsed.append((p_num, size, text))
        deduped = collapsed

    distinct_sizes = sorted({round(item[1], 1) for item in deduped}, reverse=True)[:3]
    raw_toc: list[tuple[int, str, int]] = []
    for p_num, size, text in deduped:
        rounded = round(size, 1)
        level = 1
        for lvl_idx, s_val in enumerate(distinct_sizes, start=1):
            if abs(rounded - s_val) <= 0.6:
                level = lvl_idx
                break
        else:
            level = 2
        if is_slide_deck:
            level = 1
        raw_toc.append((level, text, p_num))

    if len(raw_toc) > max(80, page_count):
        raw_toc = [item for item in raw_toc if item[0] == 1 or HEADING_NUMBERED_RE.match(item[1])]

    return assign_toc_end_pages(raw_toc[:400], page_count)


def infer_printed_offset_from_embedded_toc(
    toc: list[TocEntry],
    page_texts: list[str],
) -> int:
    """Estimate PDF-page minus printed-page offset from front-matter clues."""
    for entry in toc[:10]:
        if re.search(r"\b(chapter\s+1|introduction)\b", entry.title, re.IGNORECASE):
            text = page_texts[entry.page - 1]
            lines = [clean_whitespace(l) for l in text.splitlines() if l.strip()]
            for candidate in lines[:3] + lines[-3:]:
                if candidate.isdigit():
                    printed_num = int(candidate)
                    if 1 <= printed_num <= entry.page:
                        return entry.page - printed_num
    return 0


def infer_title(metadata: dict[str, str], first_pages: list[str], file_path: Path) -> str:
    """Resolve a human-readable book title from PDF metadata, cover text, or filename."""
    meta_title = clean_whitespace(metadata.get("title", ""))
    noisy_markers = (
        "untitled", "microsoft word", ".tex", ".dvi", ".pdf", "arxiv:",
        "f i f t h", "adaptive computation",
    )
    if (
        meta_title
        and 5 <= len(meta_title) <= 130
        and not any(m in meta_title.lower() for m in noisy_markers)
        and "_" not in meta_title
    ):
        return meta_title

    for page_text in first_pages[:2]:
        for line in page_text.splitlines():
            cleaned = clean_whitespace(line)
            if (
                8 <= len(cleaned) <= 110
                and not cleaned.lower().startswith(("arxiv", "http", "copyright", "adaptive computation"))
                and not re.match(r"^(?:f\s+i\s+f\s+t\s+h|edition)\b", cleaned, re.IGNORECASE)
            ):
                return cleaned
    return file_path.stem.replace("_", " ").strip()


def index_single_pdf(
    pdf_path: Path,
    root_dir: Path,
    existing_slugs: set[str],
) -> tuple[BookRecord, list[str], list[Counter[str]], list[Counter[str]]]:
    """Extract metadata, TOC, page text stats, and term counters from one PDF."""
    rel_path = pdf_path.relative_to(root_dir.parent).as_posix()
    rel_inside_books = pdf_path.relative_to(root_dir)
    library_category = rel_inside_books.parts[0] if len(rel_inside_books.parts) > 1 else "Uncategorized"

    sha256 = compute_sha256(pdf_path)
    file_size = pdf_path.stat().st_size
    book_slug = make_slug(pdf_path, existing_slugs)

    with pymupdf.open(pdf_path) as doc:
        page_count = doc.page_count
        raw_meta = doc.metadata or {}
        metadata = {
            key: clean_whitespace(str(val))
            for key, val in raw_meta.items()
            if val is not None and str(val).strip()
        }

        page_texts: list[str] = []
        page_labels: list[str] = []
        page_image_counts: list[int] = []
        page_unigrams: list[Counter[str]] = []
        page_bigrams: list[Counter[str]] = []

        for page_idx in range(page_count):
            page = doc[page_idx]
            text = page.get_text("text") or ""
            page_texts.append(text)
            try:
                label = page.get_label() or str(page_idx + 1)
            except Exception:
                label = str(page_idx + 1)
            page_labels.append(label)
            try:
                img_count = len(page.get_images(full=False))
            except Exception:
                img_count = 0
            page_image_counts.append(img_count)

            uni, bi = extract_tokens_and_bigrams(text)
            page_unigrams.append(Counter(uni))
            page_bigrams.append(Counter(bi))

        raw_toc = doc.get_toc(simple=True)
        toc_entries: list[TocEntry] = []
        toc_source = "none"
        printed_offset = 0

        if raw_toc:
            normalized_raw = [
                (int(item[0]), str(item[1]), int(item[2]))
                for item in raw_toc
                if len(item) >= 3 and isinstance(item[2], int) and item[2] >= 1
            ]
            if normalized_raw:
                toc_entries = assign_toc_end_pages(normalized_raw, page_count)
                toc_source = "embedded"
                printed_offset = infer_printed_offset_from_embedded_toc(toc_entries, page_texts)

        # If embedded TOC is missing or coarse (chapters-only in huge books), check printed TOC.
        if not toc_entries or (page_count >= 350 and len(toc_entries) <= 35):
            printed_toc, p_offset = infer_printed_toc(page_texts)
            if printed_toc and len(printed_toc) > len(toc_entries) * 1.5:
                toc_entries = printed_toc
                printed_offset = p_offset
                toc_source = "printed_toc" if toc_source == "none" else "embedded+printed_toc"

        if not toc_entries:
            layout_toc = infer_layout_headings(doc, page_texts)
            if layout_toc:
                toc_entries = layout_toc
                toc_source = "layout_headings"

    inferred = infer_title(metadata, page_texts[:3], pdf_path)

    pages: list[PageRecord] = []
    book_axis_totals: Counter[str] = Counter()
    for p_idx in range(page_count):
        topic_emb = compute_topic_embedding(page_unigrams[p_idx], page_bigrams[p_idx])
        for axis, weight in topic_emb.items():
            book_axis_totals[axis] += weight
        pages.append(
            PageRecord(
                page=p_idx + 1,
                label=page_labels[p_idx],
                char_count=len(page_texts[p_idx]),
                image_count=page_image_counts[p_idx],
                top_terms=[],
                topic_embedding=topic_emb,
            )
        )

    axis_norm = math.sqrt(sum(v * v for v in book_axis_totals.values()))
    domain_profile = (
        {
            axis: round(val / axis_norm, 4)
            for axis, val in sorted(book_axis_totals.items(), key=lambda item: -item[1])
        }
        if axis_norm > 0.0
        else {}
    )

    record = BookRecord(
        book_slug=book_slug,
        file_path=rel_path,
        file_name=pdf_path.name,
        library_category=library_category,
        sha256=sha256,
        file_size_bytes=file_size,
        page_count=page_count,
        metadata=metadata,
        inferred_title=inferred,
        printed_page_offset=printed_offset,
        toc_source=toc_source,
        toc=toc_entries,
        domain_profile=domain_profile,
        pages=pages,
    )

    opening_fingerprint = clean_whitespace(fold_to_ascii(" ".join(page_texts[:3])))[:2500]
    return record, [opening_fingerprint], page_unigrams, page_bigrams


def detect_duplicates(
    records: list[BookRecord],
    opening_fingerprints: list[str],
) -> list[dict[str, Any]]:
    """Identify exact SHA-256 duplicates and near-duplicate titles/editions."""
    by_hash: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_hash[rec.sha256].append(idx)

    duplicate_clusters: list[dict[str, Any]] = []

    for sha, indices in by_hash.items():
        if len(indices) <= 1:
            continue
        canonical_idx = indices[0]
        canonical_rec = records[canonical_idx]
        cluster_members = [canonical_rec.book_slug]
        for dup_idx in indices[1:]:
            dup_rec = records[dup_idx]
            dup_rec.is_duplicate = True
            dup_rec.duplicate_of = canonical_rec.book_slug
            cluster_members.append(dup_rec.book_slug)
            dup_rec.similarity_matches.append(
                DuplicateMatch(
                    other_slug=canonical_rec.book_slug,
                    other_path=canonical_rec.file_path,
                    match_type="exact_sha256",
                    title_similarity=1.0,
                    text_similarity=1.0,
                    page_count_diff=0,
                )
            )
            canonical_rec.similarity_matches.append(
                DuplicateMatch(
                    other_slug=dup_rec.book_slug,
                    other_path=dup_rec.file_path,
                    match_type="exact_sha256",
                    title_similarity=1.0,
                    text_similarity=1.0,
                    page_count_diff=0,
                )
            )
        duplicate_clusters.append(
            {
                "match_type": "exact_sha256",
                "sha256": sha,
                "canonical_slug": canonical_rec.book_slug,
                "members": cluster_members,
            }
        )

    norm_titles = [normalize_title_for_matching(r.inferred_title or r.file_name) for r in records]
    norm_files = [normalize_title_for_matching(Path(r.file_name).stem) for r in records]

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i].sha256 == records[j].sha256:
                continue
            t_sim = max(
                SequenceMatcher(None, norm_titles[i], norm_titles[j]).ratio(),
                SequenceMatcher(None, norm_files[i], norm_files[j]).ratio(),
            )
            fp_i = opening_fingerprints[i]
            fp_j = opening_fingerprints[j]
            text_sim = (
                SequenceMatcher(None, fp_i[:1200], fp_j[:1200]).ratio()
                if fp_i and fp_j
                else 0.0
            )
            page_diff = abs(records[i].page_count - records[j].page_count)
            if text_sim >= 0.85 or (t_sim >= 0.85 and page_diff <= 15):
                records[i].similarity_matches.append(
                    DuplicateMatch(
                        other_slug=records[j].book_slug,
                        other_path=records[j].file_path,
                        match_type="near_duplicate",
                        title_similarity=round(t_sim, 3),
                        text_similarity=round(text_sim, 3),
                        page_count_diff=page_diff,
                    )
                )
                records[j].similarity_matches.append(
                    DuplicateMatch(
                        other_slug=records[i].book_slug,
                        other_path=records[i].file_path,
                        match_type="near_duplicate",
                        title_similarity=round(t_sim, 3),
                        text_similarity=round(text_sim, 3),
                        page_count_diff=page_diff,
                    )
                )
                duplicate_clusters.append(
                    {
                        "match_type": "near_duplicate",
                        "members": [records[i].book_slug, records[j].book_slug],
                        "title_similarity": round(t_sim, 3),
                        "text_similarity": round(text_sim, 3),
                        "page_count_diff": page_diff,
                    }
                )

    return duplicate_clusters


def build_tfidf_and_inverted_index(
    records: list[BookRecord],
    all_page_unigrams: list[list[Counter[str]]],
    all_page_bigrams: list[list[Counter[str]]],
    max_inverted_terms: int = 3500,
) -> dict[str, dict[str, list[int]]]:
    """Compute corpus TF-IDF keywords per book/page and build an inverted index."""
    total_pages = sum(len(book_uni) for book_uni in all_page_unigrams)
    if total_pages == 0:
        return {}

    term_page_df: Counter[str] = Counter()
    term_total_tf: Counter[str] = Counter()

    for b_idx, rec in enumerate(records):
        if rec.is_duplicate:
            continue
        for p_idx in range(rec.page_count):
            uni = all_page_unigrams[b_idx][p_idx]
            bi = all_page_bigrams[b_idx][p_idx]
            seen_on_page: set[str] = set()
            for term, count in uni.items():
                if count >= 1:
                    seen_on_page.add(term)
                    term_total_tf[term] += count
            for bg, count in bi.items():
                if count >= 2 or bg in CURATED_TERMS:
                    seen_on_page.add(bg)
                    term_total_tf[bg] += count
            for term in seen_on_page:
                term_page_df[term] += 1

    max_df = max(5, int(0.45 * total_pages))
    idf: dict[str, float] = {}
    scored_vocab: list[tuple[float, str]] = []

    for term, df in term_page_df.items():
        if df < 2 and term not in CURATED_TERMS:
            continue
        if df > max_df and term not in CURATED_TERMS:
            continue
        if len(term) <= 2 and term not in {"rl", "ml", "dl", "em", "qr", "kl", "ai"}:
            continue
        term_idf = math.log((total_pages + 1.0) / (df + 1.0)) + 1.0
        idf[term] = term_idf
        curated_boost = 3.0 if term in CURATED_TERMS else (1.5 if " " in term else 1.0)
        vocab_score = curated_boost * term_total_tf[term] * term_idf
        scored_vocab.append((vocab_score, term))

    scored_vocab.sort(reverse=True)
    selected_terms: set[str] = {term for _, term in scored_vocab[:max_inverted_terms]}
    selected_terms |= {term for term in CURATED_TERMS if term in term_page_df}

    inverted_index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for b_idx, rec in enumerate(records):
        book_tfidf: Counter[str] = Counter()
        for p_idx in range(rec.page_count):
            uni = all_page_unigrams[b_idx][p_idx]
            bi = all_page_bigrams[b_idx][p_idx]
            page_scores: list[tuple[float, str]] = []

            for term, count in uni.items():
                if term in idf:
                    weight = (1.0 + math.log(count)) * idf[term]
                    page_scores.append((weight, term))
                    book_tfidf[term] += weight
                    if not rec.is_duplicate and term in selected_terms:
                        inverted_index[term][rec.book_slug].append(p_idx + 1)

            for bg, count in bi.items():
                if bg in idf:
                    weight = 1.6 * (1.0 + math.log(count)) * idf[bg]
                    page_scores.append((weight, bg))
                    book_tfidf[bg] += weight
                    if not rec.is_duplicate and bg in selected_terms:
                        inverted_index[bg][rec.book_slug].append(p_idx + 1)

            page_scores.sort(reverse=True)
            rec.pages[p_idx].top_terms = [term for _, term in page_scores[:10]]

        rec.top_keywords = [
            {"term": term, "score": round(score, 2)}
            for term, score in book_tfidf.most_common(25)
        ]

    return {
        term: {slug: pages for slug, pages in sorted(book_map.items())}
        for term, book_map in sorted(inverted_index.items())
    }


def serialize_index(
    records: list[BookRecord],
    duplicate_clusters: list[dict[str, Any]],
    inverted_index: dict[str, dict[str, list[int]]],
    errors: list[dict[str, str]],
    books_dir: Path,
) -> dict[str, Any]:
    """Assemble the final JSON-serializable book index structure."""
    total_pages = sum(r.page_count for r in records)
    unique_pages = sum(r.page_count for r in records if not r.is_duplicate)
    categories: dict[str, int] = Counter(r.library_category for r in records)

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": books_dir.as_posix(),
        "summary": {
            "total_pdfs_scanned": len(records),
            "unique_books_count": sum(1 for r in records if not r.is_duplicate),
            "exact_duplicates_count": sum(1 for r in records if r.is_duplicate),
            "near_duplicate_pairs_count": sum(
                1 for c in duplicate_clusters if c["match_type"] == "near_duplicate"
            ),
            "total_pages": total_pages,
            "unique_pages": unique_pages,
            "inverted_index_terms": len(inverted_index),
            "categories": dict(sorted(categories.items())),
            "error_count": len(errors),
        },
        "duplicates": duplicate_clusters,
        "errors": errors,
        "books": [asdict(rec) for rec in records],
        "inverted_index": inverted_index,
    }


def run_indexer(books_dir: Path, output_path: Path) -> dict[str, Any]:
    """Run the full PDF scanning, deduplication, and indexing pipeline."""
    if not books_dir.exists() or not books_dir.is_dir():
        raise FileNotFoundError(f"Books directory not found: {books_dir}")

    pdf_files = sorted(
        (p for p in books_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.as_posix().lower(),
    )
    if not pdf_files:
        raise RuntimeError(f"No PDF files discovered inside {books_dir}")

    existing_slugs: set[str] = set()
    records: list[BookRecord] = []
    opening_fingerprints: list[str] = []
    all_page_unigrams: list[list[Counter[str]]] = []
    all_page_bigrams: list[list[Counter[str]]] = []
    errors: list[dict[str, str]] = []

    for pdf_path in pdf_files:
        try:
            rec, fp_list, page_uni, page_bi = index_single_pdf(
                pdf_path=pdf_path,
                root_dir=books_dir,
                existing_slugs=existing_slugs,
            )
            records.append(rec)
            opening_fingerprints.extend(fp_list)
            all_page_unigrams.append(page_uni)
            all_page_bigrams.append(page_bi)
        except Exception as exc:  # noqa: BLE001 - capture per-file failure cleanly
            errors.append(
                {
                    "file_path": pdf_path.as_posix(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(f"[WARN] Failed to index {pdf_path}: {exc}", file=sys.stderr)

    duplicate_clusters = detect_duplicates(records, opening_fingerprints)
    inverted_index = build_tfidf_and_inverted_index(
        records=records,
        all_page_unigrams=all_page_unigrams,
        all_page_bigrams=all_page_bigrams,
    )

    payload = serialize_index(
        records=records,
        duplicate_clusters=duplicate_clusters,
        inverted_index=inverted_index,
        errors=errors,
        books_dir=books_dir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line flags for the library PDF indexer."""
    parser = argparse.ArgumentParser(
        description="Index, deduplicate, and extract TOCs/keywords from PDF textbooks."
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=Path("books_monographs"),
        help="Root folder containing raw PDF books (default: books_monographs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/book_index.json"),
        help="Destination JSON path (default: data/book_index.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        payload = run_indexer(args.books_dir, args.output)
    except Exception as exc:
        print(f"[ERROR] Indexing failed: {exc}", file=sys.stderr)
        return 1

    summary = payload["summary"]
    print(
        f"Indexed {summary['total_pdfs_scanned']} PDFs "
        f"({summary['unique_books_count']} unique, "
        f"{summary['exact_duplicates_count']} exact duplicates, "
        f"{summary['near_duplicate_pairs_count']} near-duplicate pairs), "
        f"{summary['total_pages']} total pages, "
        f"{summary['inverted_index_terms']} inverted-index terms -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
