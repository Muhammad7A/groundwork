"""Entailment checking — the second verification gate.

Grounding proves the quote is *real*. Entailment proves the quote *supports the
claim*. They are different, and the gap is a real soundness hole: a model can
quote a genuine sentence ("I use my personal ChatGPT to draft summaries") and
staple a fabricated claim to it ("the employee admitted leaking customer data").
The quote grounds perfectly; the claim is invented. Grounding alone would pass it.

So every grounded claim must also pass an entailment check: does the claim assert
anything *beyond* what the quote says? If it over-reaches — adds severity,
accusations, named parties, or facts not in the quote — it is rejected as a
semantic confabulation.

The strong version of this is a natural-language-entailment judgement that needs a
live model (``LlmEntailmentChecker``). Offline we use a deliberately *conservative*
heuristic (``HeuristicEntailmentChecker``) that catches the three mechanical
inversions — an accusation the quote never made, a polarity flip, and an invented
figure — without pretending to full NLI.

The offline checker is a floor, not a ceiling. It compares word membership and
surface cues, so a fluent paraphrase that changes scope or subject still passes.
A trained fact-checking model (MiniCheck, AlignScore) drops in behind the same
``EntailmentChecker`` Protocol and is the intended production answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..lexicon import invents_quantity, negation_disagrees
from ..llm.client import LLMClient
from ..transcript.model import Transcript
from .model import Claim


class Entailment(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"


@dataclass(frozen=True)
class EntailmentResult:
    verdict: Entailment
    reason: str
    method: str  # "heuristic" | "llm"


@dataclass
class EntailmentReject:
    claim: Claim
    reason: str


@runtime_checkable
class EntailmentChecker(Protocol):
    def check(self, statement: str, quote: str) -> EntailmentResult: ...


# --- the conservative offline heuristic -----------------------------------

# Accusatory / legal stems. If a claim asserts one of these and the quote does
# NOT contain it, the claim has escalated beyond its evidence — the dangerous,
# legally-radioactive case (F5). Kept narrow to avoid false positives on ordinary
# words (e.g. we do NOT include generic terms like "data" or "customer").
# Two entries were wider than their meaning and rejected honest claims:
#
#   "forg"        fired on forgot / forget / forgetting / forgive — the single most
#                 ordinary thing an interviewee says about a missed step. Narrowed
#                 to the forms that actually allege document forgery. ("forge" is
#                 no good either: it is inside "forget".)
#   "confidential" fired on "this is confidential" and "we signed a confidentiality
#                 agreement", which are routine business vocabulary rather than
#                 accusations. Dropped; a real disclosure incident is already
#                 covered by "leak" and "breach".
#
# A false reject here is not free: it silently drops a true finding, and the
# rejection reason blames the interviewee for asserting something they did not.
_SEVERE_STEMS = (
    "illeg", "fraud", "leak", "stol", "steal", "theft", "embezzl", "brib",
    "kickback", "harass", "discrimin", "lawsuit", "misconduct", "breach",
    "violat", "sabotag", "launder", "forger", "forged", "coverup",
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "is",
    "are", "was", "were", "be", "been", "it", "its", "that", "this", "with",
    "as", "at", "by", "so", "we", "i", "they", "he", "she", "you", "my", "our",
    "their", "his", "her", "them", "about", "because", "than", "then", "just",
    "really", "very", "some", "any", "all", "not", "no", "do", "does", "did",
    "have", "has", "had", "from", "up", "out", "into", "over", "before", "after",
}

_MIN_OVERLAP = 0.30


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def _severe_stem(word: str) -> str | None:
    for stem in _SEVERE_STEMS:
        if stem in word:
            return stem
    return None


class HeuristicEntailmentChecker:
    """Offline, conservative. Rejects escalation, polarity inversion, invented
    quantities, and near-zero overlap.

    It is a **smoke test, not a substitute for entailment.** It reasons about word
    membership, polarity cues and figures — never about meaning — so a paraphrase
    that reverses a claim's scope or subject will pass it. Treat a SUPPORTED
    verdict from this checker as "no mechanical inversion detected", not as
    "verified". Where a real judgement is required, supply an
    :class:`LlmEntailmentChecker` (or a trained NLI model behind the same
    Protocol) via :func:`make_checker`.
    """

    def check(self, statement: str, quote: str) -> EntailmentResult:
        claim_words = _content_words(statement)
        quote_words = _content_words(quote)

        # 1. Escalation: a serious/accusatory term in the claim, absent from the quote.
        for word in claim_words:
            stem = _severe_stem(word)
            if stem and not any(stem in qw for qw in quote_words):
                return EntailmentResult(
                    Entailment.NOT_SUPPORTED,
                    f"claim asserts '{word}' with no basis in the quote",
                    "heuristic",
                )

        # 2. Polarity inversion. Word overlap cannot see negation, so "I don't
        # think the approval step is a problem" and "the approval step is a
        # problem" scored as a near-perfect match — the claim asserting the
        # opposite of its own evidence, carrying a genuine verbatim quote. That
        # is the worst output this system can produce, so it is checked before
        # overlap rather than after.
        if negation_disagrees(statement, quote):
            return EntailmentResult(
                Entailment.NOT_SUPPORTED,
                "claim and quote disagree in polarity (one negates, the other does not)",
                "heuristic",
            )

        # 3. Invented quantity. Content words are letters-only, so every figure —
        # a duration, a headcount, a cost — passed unchecked and "three days"
        # could become "thirty days" with the overlap barely moving. Numbers are
        # what become business cases downstream, so a figure the quote does not
        # contain is a fabrication regardless of how well the words match.
        invented = invents_quantity(statement, quote)
        if invented is not None:
            return EntailmentResult(
                Entailment.NOT_SUPPORTED,
                f"claim asserts the quantity '{invented}', which is not in the quote",
                "heuristic",
            )

        # 4. Low overlap — with two honest rescues before rejection. A live model
        # legitimately writes claims that SUMMARY the quote (adding context words
        # that lower the ratio) or TRIM it (keeping only the load-bearing words:
        # "sometimes I'm probably underutilized about a day a week" summarized as
        # "about a day a week lost to waiting"). Both are faithful restatements;
        # neither is a fabrication. The rescues, in order of strictness:
        #
        #   a) QUOTE SUBSET OF CLAIM (word-set): the claim carries every content
        #      word of the quote — it added, never swapped.
        #   b) QUOTE EMBEDDED IN CLAIM (verbatim string, punctuation-tolerant):
        #      the claim literally contains the quote's words — the model quoted
        #      inside its sentence and the overlap metric punished shared words
        #      on the quote's side only. A fabricated claim cannot pass either
        #      rescue without actually carrying the quote's words, and every
        #      rejection gate above (escalation, polarity, invented quantity)
        #      still ran first.
        if claim_words:
            overlap = len(claim_words & quote_words) / len(claim_words)
            if overlap < _MIN_OVERLAP:
                covers_quote = bool(quote_words) and quote_words <= claim_words
                embedded = _quote_embedded(statement, quote)
                if not covers_quote and not embedded:
                    return EntailmentResult(
                        Entailment.NOT_SUPPORTED,
                        f"claim overlaps the quote only {overlap:.0%}",
                        "heuristic",
                    )
                return EntailmentResult(
                    Entailment.SUPPORTED,
                    "claim is a faithful restatement "
                    + ("(contains the quote's full content) " if covers_quote else "")
                    + ("(embeds the quote verbatim) " if embedded else "")
                    + "— no escalation, polarity flip, or invented quantity",
                    "heuristic",
                )
        return EntailmentResult(Entailment.SUPPORTED, "claim stays within the quote", "heuristic")


def _quote_embedded(statement: str, quote: str) -> bool:
    """Whether the quote's words appear in the claim as a contiguous-enough
    run. Punctuation/case tolerant (the same normalization grounding uses),
    word-subsequence at the claim level: the quote's words, in order, must all
    be present in the claim. A claim about something else cannot contain them."""
    claim_norm = re.sub(r"[^\w\s]", " ", statement.lower()).split()
    quote_norm = re.sub(r"[^\w\s]", " ", quote.lower()).split()
    if not quote_norm:
        return False
    it = iter(claim_norm)
    return all(any(word == c for c in it) for word in quote_norm)


# --- the live model checker ------------------------------------------------

ENTAILMENT_PROMPT_VERSION = "entailment/v0.1"

ENTAILMENT_SYSTEM = """\
You are a strict evidence auditor. You are given a QUOTE (verbatim words an \
employee said) and a CLAIM about it. Decide whether the quote SUPPORTS the claim.

Answer SUPPORTED only if the claim asserts nothing beyond what the quote says or \
plainly implies. Answer NOT_SUPPORTED if the claim adds ANY of: severity or \
accusation not in the quote, named people/teams not in the quote, specific numbers \
not in the quote, or facts the quote does not establish. When in doubt, answer \
NOT_SUPPORTED — an unsupported claim is worse than a missing one.

Return ONLY a JSON object: {"verdict": "SUPPORTED" | "NOT_SUPPORTED", "reason": "short"}.
"""


class LlmEntailmentChecker:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def check(self, statement: str, quote: str) -> EntailmentResult:
        text = self._llm.complete(
            system=ENTAILMENT_SYSTEM,
            messages=[{"role": "user", "content": f'QUOTE: "{quote}"\nCLAIM: "{statement}"'}],
            max_tokens=200,
            temperature=0.0,
        )
        data = _extract_json(text)
        if not data:
            # Fail closed: a grounded claim we could not verify is dropped, not passed.
            return EntailmentResult(Entailment.NOT_SUPPORTED, "unverifiable response", "llm")
        verdict = str(data.get("verdict", "")).strip().upper()
        reason = str(data.get("reason", ""))
        if verdict == "SUPPORTED":
            return EntailmentResult(Entailment.SUPPORTED, reason, "llm")
        return EntailmentResult(Entailment.NOT_SUPPORTED, reason or "not supported", "llm")


def make_checker(llm: LLMClient | None) -> EntailmentChecker:
    return LlmEntailmentChecker(llm) if llm is not None else HeuristicEntailmentChecker()


def apply_entailment(
    claims: list[Claim], transcript: Transcript, checker: EntailmentChecker
) -> tuple[list[Claim], list[EntailmentReject]]:
    """Second gate: keep only claims whose evidence supports them.

    **Every** piece of evidence is checked, not just the first. Grounding emits one
    piece per claim today, so reading ``evidence[0]`` was correct in practice and
    silently wrong in principle: the moment a claim carries corroborating evidence,
    the unchecked pieces would ride in behind the checked one, and the gate would
    report a verdict it had not actually reached.

    A claim survives if *any* piece of its evidence supports it — corroboration
    means several quotes bear on one statement and only one need establish it —
    but the rejection reason names how many were tried, so a claim rejected against
    four quotes does not read like a claim rejected against one.
    """
    kept: list[Claim] = []
    rejected: list[EntailmentReject] = []
    for claim in claims:
        reasons: list[str] = []
        supported = False
        for evidence in claim.evidence:
            result = checker.check(claim.statement, evidence.resolve(transcript))
            if result.verdict is Entailment.SUPPORTED:
                supported = True
                break
            reasons.append(result.reason)
        if supported:
            kept.append(claim)
        elif reasons:
            detail = reasons[0] if len(reasons) == 1 else (
                f"{reasons[0]} (and {len(reasons) - 1} further piece(s) of "
                f"evidence did not support it either)")
            rejected.append(EntailmentReject(claim=claim, reason=detail))
        else:
            rejected.append(EntailmentReject(
                claim=claim, reason="the claim carries no evidence to check"))
    return kept, rejected


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
