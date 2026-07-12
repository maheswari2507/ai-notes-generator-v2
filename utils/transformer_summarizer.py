import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

STOP_WORDS = set(stopwords.words("english"))


def _clean(text):
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sentence_scores(sentences):
    words = word_tokenize(" ".join(sentences).lower())

    freq = Counter(
        word for word in words
        if word.isalnum() and word not in STOP_WORDS
    )

    scores = {}

    for sentence in sentences:
        score = 0
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                score += freq[word]
        scores[sentence] = score

    return scores


def _overall_summary(sentences):
    scores = _sentence_scores(sentences)

    best = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return " ".join([x[0] for x in best])


def _paragraphs(sentences):
    paras = []

    for i in range(0, len(sentences), 3):
        para = " ".join(sentences[i:i + 3])
        if para.strip():
            paras.append(para)

    return paras


def _key_points(sentences):
    scores = _sentence_scores(sentences)

    best = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return [x[0] for x in best]


def generate_summary(text):

    text = _clean(text)

    sentences = sent_tokenize(text)

    if len(sentences) == 0:
        return {
            "overall_summary": "",
            "paragraphs": [],
            "summary": "",
            "key_points": []
        }

    overall = _overall_summary(sentences)

    paragraphs = _paragraphs(sentences)

    key_points = _key_points(sentences)

    return {
        "overall_summary": overall,
        "paragraphs": paragraphs,
        "summary": overall,
        "key_points": key_points
    }