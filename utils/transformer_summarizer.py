import re
import math
from transformers import pipeline

print("Loading models...")
summarizer = pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    framework="pt"
)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

WORDS_PER_PAGE  = 250
POINTS_PER_PAGE = 2
MIN_POINTS      = 5
MAX_POINTS      = 60
CHUNK_WORDS     = 300


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove PDF artifacts and normalize whitespace."""
    text = re.sub(r'\(cid:\d+\)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _target_points(word_count: int) -> int:
    """Dynamically calculate how many key points to generate based on document size."""
    pages = max(1, word_count / WORDS_PER_PAGE)
    return max(MIN_POINTS, min(MAX_POINTS, round(pages * POINTS_PER_PAGE)))


def _split_into_word_chunks(text: str, chunk_words: int) -> list:
    """Split text into word-count-bounded chunks, skipping very short ones."""
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i: i + chunk_words])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
    return chunks


def _chunk_to_sentences(text: str) -> list:
    """Split text into sentences, filtering out very short or empty ones."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if len(s.split()) >= 6]


def _split_into_short_paragraphs(text: str, max_sentences: int = 3) -> list:
    """Break a long block of text into groups of max_sentences sentences."""
    sentences = _chunk_to_sentences(text)
    paras = []
    for i in range(0, len(sentences), max_sentences):
        group = sentences[i: i + max_sentences]
        if group:
            paras.append(" ".join(group))
    return paras if paras else [text]


# ─────────────────────────────────────────────
#  Summarization
# ─────────────────────────────────────────────

def _summarize_chunk(chunk: str, max_len: int, min_len: int) -> str:
    """Summarize a single chunk of text using DistilBART."""
    try:
        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        text = result[0]["summary_text"].strip()
        # Remove leading bullet/dash artifacts
        text = re.sub(r'^[\s\-–—•*]+', '', text).strip()
        return text
    except Exception as e:
        print(f"  Warning: summarization failed ({e}), using fallback")
        sents = _chunk_to_sentences(chunk)
        return " ".join(sents[:3])


def _generate_overall_summary(chunks: list) -> str:
    """
    Generate a short TL;DR overview from the first and last chunks.
    Length scales slightly with number of chunks.
    """
    combined   = chunks[0] + " " + (chunks[-1] if len(chunks) > 1 else "")
    word_count = len(combined.split())
    # Scale max_len slightly with document size
    max_len    = min(100, max(40, word_count // 4))
    return _summarize_chunk(combined[:1500], max_len, min_len=25)


def _generate_full_summary(text: str) -> list:
    """
    Summarize each chunk independently, then break each chunk summary
    into short paragraphs of max 3 sentences each.
    Returns a flat list of paragraph strings.
    """
    chunks = _split_into_word_chunks(text, CHUNK_WORDS)
    print(f"  Summary chunks: {len(chunks)}")

    all_paragraphs = []

    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i+1}/{len(chunks)}...")

        word_count = len(chunk.split())
        max_len    = min(150, max(60, word_count // 2))
        min_len    = max(40, max_len // 2)

        raw = _summarize_chunk(chunk, max_len, min_len)

        # Break into short paragraphs (max 3 sentences each)
        short_paras = _split_into_short_paragraphs(raw, max_sentences=3)
        all_paragraphs.extend(short_paras)

    print(f"  Total short paragraphs: {len(all_paragraphs)}")
    return all_paragraphs


# ─────────────────────────────────────────────
#  Key Point Extraction (DistilBART only)
# ─────────────────────────────────────────────

def _is_complete_sentence(sentence: str) -> bool:
    """
    Check if a sentence looks complete:
    - Ends with proper punctuation
    - Has a minimum number of words
    - Does not start with a lowercase continuation word
    """
    sentence = sentence.strip()
    if not sentence:
        return False
    # Must end with sentence-ending punctuation
    if not re.search(r'[.!?]$', sentence):
        return False
    # Must have at least 5 words
    if len(sentence.split()) < 5:
        return False
    # Reject sentences starting mid-thought (e.g. "and", "but", "or", "which")
    first_word = sentence.split()[0].lower()
    if first_word in {"and", "but", "or", "which", "that", "however", "also", "while"}:
        return False
    return True


def _deduplicate_sentences(sentences: list) -> list:
    """
    Remove near-duplicate sentences using simple word-overlap ratio.
    Also removes exact duplicates.
    """
    unique   = []
    seen_set = set()

    for s in sentences:
        normalized = re.sub(r'\s+', ' ', s.lower().strip())
        if normalized in seen_set:
            continue

        # Check overlap with already accepted sentences
        words_s = set(normalized.split())
        is_dup  = False
        for accepted in unique:
            words_a  = set(re.sub(r'\s+', ' ', accepted.lower().strip()).split())
            if not words_s or not words_a:
                continue
            overlap = len(words_s & words_a) / max(len(words_s), len(words_a))
            if overlap >= 0.75:
                is_dup = True
                break

        if not is_dup:
            seen_set.add(normalized)
            unique.append(s)

    return unique


def _extract_key_points(paragraphs: list, num_points: int) -> list:
    """
    Extract key points directly from the already-generated summary paragraphs.
    No additional model calls. No KeyBERT. No SentenceTransformer.

    Steps:
      1. Collect all sentences from paragraph summaries.
      2. Filter out incomplete or very short sentences.
      3. Deduplicate near-similar sentences.
      4. Take the top num_points in original order.
    """
    all_sentences = []

    for para in paragraphs:
        sentences = _chunk_to_sentences(para)
        all_sentences.extend(sentences)

    print(f"  Total raw candidate sentences: {len(all_sentences)}")

    # Filter: keep only complete, meaningful sentences
    filtered = [s for s in all_sentences if _is_complete_sentence(s)]
    print(f"  After completeness filter: {len(filtered)}")

    # Deduplicate
    deduped = _deduplicate_sentences(filtered)
    print(f"  After deduplication: {len(deduped)}")

    # Take up to num_points in preserved order
    key_points = deduped[:num_points]
    print(f"  Final key points selected: {len(key_points)}")

    return key_points


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def generate_summary(text: str) -> dict:
    """
    Main function called by the Flask app.
    Returns a dict with keys:
      - overall_summary : short TL;DR string
      - paragraphs      : list of short paragraph strings
      - summary         : alias for overall_summary (backwards compat)
      - key_points      : list of key point strings
    """
    text       = _clean(text)
    word_count = len(text.split())
    print(f"\nWORD COUNT: {word_count}")

    num_points = _target_points(word_count)
    print(f"TARGET KEY POINTS: {num_points}")

    chunks = _split_into_word_chunks(text, CHUNK_WORDS)

    # Step 1: Short TL;DR overview
    print("\n── Generating overall summary...")
    overall_summary = _generate_overall_summary(chunks)

    # Step 2: Per-chunk detailed paragraphs
    print("\n── Generating detailed paragraphs...")
    paragraphs = _generate_full_summary(text)

    # Step 3: Key points extracted from paragraphs — zero extra model calls
    print("\n── Extracting key points...")
    key_points = _extract_key_points(paragraphs, num_points)

    print(f"\nOVERALL SUMMARY: 1 paragraph")
    print(f"DETAIL PARAGRAPHS: {len(paragraphs)}")
    print(f"KEY POINTS: {len(key_points)}")

    return {
        "overall_summary": overall_summary,
        "paragraphs":      paragraphs,
        "summary":         overall_summary,
        "key_points":      key_points,
    }
