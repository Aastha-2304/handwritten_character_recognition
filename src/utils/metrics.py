"""
Evaluation metrics for Handwritten Text Recognition:
Character Error Rate (CER) and Word Error Rate (WER).
Computes Levenshtein edit distance between ground-truth and predictions.
"""
from __future__ import annotations


def levenshtein_distance(seq1: list | str, seq2: list | str) -> int:
    """
    Computes the minimum edit distance (insertions, deletions, substitutions)
    between two sequences using standard dynamic programming.
    """
    len1, len2 = len(seq1), len(seq2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[len1][len2]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    CER = LevenshteinDistance(reference, hypothesis) / len(reference)
    Returns a float in [0.0, infinity) where 0.0 means perfect prediction.
    """
    ref_len = len(reference)
    if ref_len == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    dist = levenshtein_distance(reference, hypothesis)
    return dist / float(ref_len)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    WER = LevenshteinDistance(reference_words, hypothesis_words) / len(reference_words)
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / float(len(ref_words))
