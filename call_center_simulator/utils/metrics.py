"""Evaluation metrics for the call-center simulator."""

from __future__ import annotations

import itertools
import math
from collections import Counter

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer


def compute_mape_ocean(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, list[float]]:
    """Compute MAPE averaged over OCEAN axes and batch.

    Args:
        preds: Predicted OCEAN values, shape [B, 5], values in [0, 1].
        targets: Target OCEAN values, shape [B, 5], values in [0, 1].
        eps: Small constant to avoid division by zero.

    Returns:
        Tuple of (mean_mape, per_axis_mape). Axis order: O, C, E, A, N.
    """
    abs_err = (preds - targets).abs()
    denom = targets.abs().clamp(min=eps)
    per_sample_per_axis = abs_err / denom  # [B, 5]
    per_axis = per_sample_per_axis.mean(dim=0).tolist()  # [5]
    mean_mape = float(sum(per_axis) / len(per_axis))
    return mean_mape, per_axis


def compute_perplexity(avg_cross_entropy_loss: float) -> float:
    """Compute perplexity from average cross-entropy loss.

    Args:
        avg_cross_entropy_loss: Mean token-level cross-entropy loss (nats).

    Returns:
        Perplexity = exp(loss).
    """
    return math.exp(avg_cross_entropy_loss)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus BLEU-4 score.

    Args:
        hypotheses: List of generated strings.
        references: List of reference strings (one per hypothesis).

    Returns:
        BLEU-4 score in [0, 1].
    """
    tokenized_hyps = [h.split() for h in hypotheses]
    tokenized_refs = [[r.split()] for r in references]
    smoothing = SmoothingFunction().method1
    return float(
        corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothing)
    )


def compute_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Compute mean ROUGE-L F1 score.

    Args:
        hypotheses: List of generated strings.
        references: List of reference strings.

    Returns:
        Mean ROUGE-L F1 in [0, 1].
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references, strict=False)
    ]
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_distinct(texts: list[str]) -> tuple[float, float]:
    """Compute Distinct-1 and Distinct-2 (anti-collapse diversity metrics).

    Args:
        texts: List of generated strings.

    Returns:
        Tuple of (distinct_1, distinct_2).
        distinct_1 = unique_unigrams / total_unigrams
        distinct_2 = unique_bigrams / total_bigrams
    """
    all_tokens: list[str] = []
    for text in texts:
        all_tokens.extend(text.split())

    if not all_tokens:
        return 0.0, 0.0

    distinct_1 = len(Counter(all_tokens)) / len(all_tokens)

    bigrams = list(itertools.pairwise(all_tokens))
    if not bigrams:
        return distinct_1, 0.0

    distinct_2 = len(Counter(bigrams)) / len(bigrams)
    return distinct_1, distinct_2
