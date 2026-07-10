from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


def generate_summary(text):

    stop_words = set(stopwords.words('english'))

    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Build word frequencies
    word_frequencies = {}

    for word in words:
        if word.isalnum() and word not in stop_words:
            word_frequencies[word] = (
                word_frequencies.get(word, 0) + 1
            )

    # Handle empty input
    if not word_frequencies:
        return {
            "summary": "No meaningful text found.",
            "key_points": []
        }

    # Normalize frequencies
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    sentence_scores = {}

    # Score each sentence
    for sentence in sentences:

        sentence_words = word_tokenize(
            sentence.lower()
        )

        sentence_length = len(sentence_words)

        # Ignore very short and very long sentences
        if sentence_length < 8 or sentence_length > 30:
            continue

        # Ignore examples and exercises
        bad_words = [
            "example",
            "examples",
            "activity",
            "answer key",
            "question",
            "answer",
            "i ",
            "you ",
            "we "
        ]

        if any(
            bad_word in sentence.lower()
            for bad_word in bad_words
        ):
            continue

        score = 0

        for word in sentence_words:
            if word in word_frequencies:
                score += word_frequencies[word]

        # Prevent long sentences from dominating
        score = score / sentence_length

        # Bonus for educational keywords
        important_words = [
            "important",
            "definition",
            "conclusion",
            "advantage",
            "disadvantage",
            "process",
            "types",
            "classification",
            "features",
            "components",
            "uses",
            "applications",
            "article",
            "articles",
            "noun",
            "specific",
            "general",
            "indefinite",
            "definite",
            "rule"
        ]

        for keyword in important_words:
            if keyword in sentence.lower():
                score *= 1.5

        sentence_scores[sentence] = score

    # Sort sentences
    ranked_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )

    word_count = len(words)

    # Dynamic summary size
    if word_count <= 100:
        summary_count = 2
    elif word_count <= 300:
        summary_count = 3
    elif word_count <= 700:
        summary_count = 5
    else:
        summary_count = 8

    # Remove duplicates
    summary_sentences = []

    for sentence in ranked_sentences:

        duplicate = False

        for existing in summary_sentences:

            common_words = (
                set(sentence.lower().split())
                &
                set(existing.lower().split())
            )

            if len(common_words) > 6:
                duplicate = True
                break

        if not duplicate:
            summary_sentences.append(sentence)

        if len(summary_sentences) >= summary_count:
            break

    # Preserve original order
    summary_sentences = sorted(
        summary_sentences,
        key=lambda x: sentences.index(x)
    )

    summary = "\n\n".join(summary_sentences)

    return {
        "summary": summary,
        "key_points": summary_sentences
    }