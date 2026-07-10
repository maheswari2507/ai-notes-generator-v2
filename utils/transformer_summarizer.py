import re
import math
from transformers import pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading models...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
kw_model   = KeyBERT()
encoder    = SentenceTransformer("all-MiniLM-L6-v2")

WORDS_PER_PAGE   = 250
POINTS_PER_PAGE  = 2
MIN_POINTS       = 5
MAX_POINTS       = 60
SIM_THRESHOLD    = 0.80
CHUNK_WORDS      = 300


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r'\(cid:\d+\)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _target_points(word_count: int) -> int:
    pages = max(1, word_count / WORDS_PER_PAGE)
    return max(MIN_POINTS, min(MAX_POINTS, round(pages * POINTS_PER_PAGE)))


def _split_into_word_chunks(text: str, chunk_words: int) -> list[str]:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i: i + chunk_words])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
    return chunks


def _chunk_to_sentences(chunk_text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', chunk_text)
    return [s.strip() for s in parts if len(s.split()) >= 6]


def _split_into_short_paragraphs(text: str, max_sentences: int = 3) -> list[str]:
    """Break a long paragraph into groups of max_sentences sentences."""
    sentences = _chunk_to_sentences(text)
    paras = []
    for i in range(0, len(sentences), max_sentences):
        group = sentences[i: i + max_sentences]
        if group:
            paras.append(" ".join(group))
    return paras if paras else [text]


# ─────────────────────────────────────────────
#  Summary generation
# ─────────────────────────────────────────────

def _summarize_chunk(chunk: str, max_len: int, min_len: int) -> str:
    """Summarize one chunk, return cleaned string."""
    try:
        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        text = result[0]["summary_text"].strip()
        text = re.sub(r'^[\s\-–—•*]+', '', text).strip()
        return text
    except Exception as e:
        print(f"  Warning: summarization failed ({e}), using fallback")
        sents = _chunk_to_sentences(chunk)
        return " ".join(sents[:3])


def _generate_full_summary(text: str) -> list[str]:
    """
    Summarize each chunk independently, then break each chunk summary
    into short paragraphs of max 3 sentences.
    Returns a flat list of short paragraph strings.
    """
    chunks = _split_into_word_chunks(text, CHUNK_WORDS)
    print(f"  Summary chunks: {len(chunks)}")

    all_paragraphs = []

    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i+1}/{len(chunks)}...")

        word_count = len(chunk.split())
        max_len    = min(150, max(60, word_count // 2))
        min_len    = max(40,  max_len // 2)

        raw = _summarize_chunk(chunk, max_len, min_len)

        # Break into short paragraphs (max 3 sentences each)
        short_paras = _split_into_short_paragraphs(raw, max_sentences=3)
        all_paragraphs.extend(short_paras)

    print(f"  Total short paragraphs: {len(all_paragraphs)}")
    return all_paragraphs


def _generate_overall_summary(chunks: list[str]) -> str:
    """
    Short 2-3 sentence TL;DR from first + last chunk.
    """
    combined   = chunks[0] + " " + (chunks[-1] if len(chunks) > 1 else "")
    word_count = len(combined.split())
    max_len    = min(80, max(40, word_count // 4))
    return _summarize_chunk(combined[:1500], max_len, min_len=25)


# ─────────────────────────────────────────────
#  Key point extraction
# ─────────────────────────────────────────────

def _summarize_chunk_to_sentences(chunk: str) -> list[str]:
    """For key point extraction — sentences from chunk summary."""
    word_count = len(chunk.split())
    max_len    = min(120, max(40, word_count // 3))
    min_len    = min(30,  max_len // 3)
    try:
        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        return _chunk_to_sentences(result[0]["summary_text"])
    except Exception:
        return _chunk_to_sentences(chunk)[:3]


def _mmr_select(
    candidates: list[str],
    embeddings: np.ndarray,
    kw_emb: np.ndarray,
    num_points: int,
) -> list[int]:
    relevance  = cosine_similarity(embeddings, kw_emb.reshape(1, -1)).flatten()
    sim_matrix = cosine_similarity(embeddings)
    ranked     = sorted(range(len(candidates)),
                        key=lambda i: relevance[i], reverse=True)

    selected: list[int] = []
    for idx in ranked:
        if len(selected) >= num_points:
            break
        if not any(sim_matrix[idx][s] >= SIM_THRESHOLD for s in selected):
            selected.append(idx)

    return sorted(selected)


def _extract_key_points(
    text: str,
    num_points: int,
    paragraphs: list[str],
) -> list[str]:
    """
    Candidate pool:
      (a) sentences from already-computed summary paragraphs  — no extra model calls
      (b) raw sentences from each chunk for extra coverage
    """
    chunks = _split_into_word_chunks(text, CHUNK_WORDS)

    all_candidates: list[str] = []

    # (a) reuse summary paragraphs
    for para in paragraphs:
        all_candidates.extend(_chunk_to_sentences(para))

    # (b) raw sentences
    for i, chunk in enumerate(chunks):
        raw_sents = _chunk_to_sentences(chunk)
        all_candidates.extend(raw_sents)
        print(f"  Chunk {i+1}/{len(chunks)}: {len(raw_sents)} raw sentences")

    # Exact-string dedup
    seen: set[str] = set()
    unique: list[str] = []
    for s in all_candidates:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    print(f"  Total unique candidates: {len(unique)}")
    if not unique:
        return []

    kw_results  = kw_model.extract_keywords(
        text[:8000],
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=25,
    )
    keyword_str = " ".join(kw for kw, _ in kw_results)

    all_embs  = encoder.encode(
        unique + [keyword_str],
        show_progress_bar=False,
        batch_size=64,
    )
    cand_embs = all_embs[:-1]
    kw_emb    = all_embs[-1]

    selected = _mmr_select(unique, cand_embs, kw_emb, num_points)
    print(f"  After MMR: {len(selected)} key points")
    return [unique[i] for i in selected]


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def generate_summary(text: str) -> dict:
    text       = _clean(text)
    word_count = len(text.split())
    print(f"\nWORD COUNT: {word_count}")

    num_points = _target_points(word_count)
    print(f"TARGET KEY POINTS: {num_points}")

    chunks = _split_into_word_chunks(text, CHUNK_WORDS)

    # Short TL;DR at the top
    print("\n── Generating overall summary...")
    overall_summary = _generate_overall_summary(chunks)

    # Per-chunk short paragraphs
    print("\n── Generating detailed paragraphs...")
    paragraphs = _generate_full_summary(text)

    # Key points — reuses paragraphs, no extra summarizer calls
    print("\n── Extracting key points...")
    key_points = _extract_key_points(text, num_points, paragraphs)

    print(f"\nOVERALL SUMMARY: 1 paragraph")
    print(f"DETAIL PARAGRAPHS: {len(paragraphs)}")
    print(f"KEY POINTS: {len(key_points)}")

    return {
        "overall_summary": overall_summary,   # short TL;DR string
        "paragraphs":      paragraphs,         # list of short paragraph strings
        "summary":         overall_summary,    # backwards compat alias
        "key_points":      key_points,
    }