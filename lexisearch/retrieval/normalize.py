"""Query normalisation and preprocessing pipeline.

Transforms raw query strings into clean, canonical forms before they
reach any retrieval component.  The normalisation steps are composable:
each step is an independent callable so they can be combined, reordered,
or swapped out without touching application code.

Typical flow::

    from lexisearch.retrieval.normalize import QueryNormalizer, NormalizerConfig

    cfg = NormalizerConfig(
        lowercase=True,
        remove_stopwords=True,
        stopwords={"the", "a", "an", "of", "in"},
        max_tokens=64,
    )
    normalizer = QueryNormalizer(cfg)

    result = normalizer.normalize("  What ARE  the  BEST  approaches  for IR? ")
    print(result.normalized)   # "best approaches ir"
    print(result.removed_stopwords)  # ['what', 'are', 'the', 'the', 'for']
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Language detection heuristics
# ---------------------------------------------------------------------------

# Simple script-range checks — no external dependency required.
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")


def detect_script(text: str) -> str:
    """Heuristically detect the dominant script family in *text*.

    Returns one of ``"latin"``, ``"cjk"``, ``"arabic"``, ``"cyrillic"``,
    or ``"unknown"``.  Only the character counts of the four supported
    families are compared; the largest wins.
    """
    counts: dict[str, int] = {
        "latin": len(_LATIN_RE.findall(text)),
        "cjk": len(_CJK_RE.findall(text)),
        "arabic": len(_ARABIC_RE.findall(text)),
        "cyrillic": len(_CYRILLIC_RE.findall(text)),
    }
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else "unknown"


# ---------------------------------------------------------------------------
# Built-in stopword sets
# ---------------------------------------------------------------------------

#: Conservative English stopwords — frequently occurring function words that
#: carry little discriminative value in keyword search.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "if",
        "into",
        "about",
        "above",
        "after",
        "before",
        "between",
        "during",
        "through",
        "while",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "because",
        "also",
        "only",
        "own",
        "same",
        "then",
        "there",
        "their",
        "they",
        "them",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "him",
        "her",
        "my",
        "me",
        "i",
        "up",
        "out",
        "over",
        "under",
        "again",
        "further",
        "here",
        "once",
    }
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class PunctuationPolicy(Enum):
    """How to handle punctuation during normalisation.

    Attributes:
        REMOVE: Strip all punctuation characters.
        REPLACE_WITH_SPACE: Replace punctuation with a single space.
        KEEP: Leave punctuation untouched.
    """

    REMOVE = "remove"
    REPLACE_WITH_SPACE = "replace_with_space"
    KEEP = "keep"


@dataclass
class NormalizerConfig:
    """Configuration for :class:`QueryNormalizer`.

    Attributes:
        lowercase: Convert the query to lower-case before any other step.
        unicode_normalize: Apply Unicode NFC normalisation to collapse
            composed / decomposed character variants.
        expand_contractions: Expand common English contractions
            (e.g. ``"don't"`` → ``"do not"``).
        punctuation_policy: How to handle punctuation (see
            :class:`PunctuationPolicy`).
        collapse_whitespace: Collapse runs of whitespace into a single
            space and strip leading/trailing whitespace.
        remove_stopwords: Remove tokens that appear in *stopwords*.
        stopwords: Set of stopwords to remove.  Defaults to
            :data:`ENGLISH_STOPWORDS` when *None*.
        min_token_length: Discard tokens shorter than this value (0 = keep all).
        max_tokens: Truncate the token list to this many tokens after all
            other steps (0 = no limit).
        min_query_length: If the final normalised string is shorter than
            this, :meth:`QueryNormalizer.normalize` marks the result as
            *too_short*.
        preserve_quoted_phrases: Leave tokens inside ``"..."`` together
            as a single unit and skip stopword removal for them.
        extra_steps: Additional callables applied to the token list in
            order.  Each callable receives ``list[str]`` and returns
            ``list[str]``.
    """

    lowercase: bool = True
    unicode_normalize: bool = True
    expand_contractions: bool = True
    punctuation_policy: PunctuationPolicy = PunctuationPolicy.REPLACE_WITH_SPACE
    collapse_whitespace: bool = True
    remove_stopwords: bool = False
    stopwords: frozenset[str] | set[str] | None = None
    min_token_length: int = 1
    max_tokens: int = 0
    min_query_length: int = 2
    preserve_quoted_phrases: bool = True
    extra_steps: list[Callable[[list[str]], list[str]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Apply default stopwords when none supplied explicitly."""
        if self.stopwords is None:
            self.stopwords = ENGLISH_STOPWORDS


# ---------------------------------------------------------------------------
# Contraction map
# ---------------------------------------------------------------------------

_CONTRACTIONS: dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "mightn't": "might not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "how's": "how is",
    "let's": "let us",
}

# Build a single regex from all contraction keys (longest match first).
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _replace_contraction(match: re.Match[str]) -> str:
    token = match.group(0).lower()
    return _CONTRACTIONS.get(token, token)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Output from :meth:`QueryNormalizer.normalize`.

    Attributes:
        original: The raw input query.
        normalized: The fully processed query string ready for retrieval.
        tokens: The token list after all normalisation steps (before joining).
        removed_stopwords: Stopword tokens that were removed (order preserved).
        truncated: ``True`` if the token list was truncated at *max_tokens*.
        too_short: ``True`` if the normalised string is shorter than
            *min_query_length* (may indicate a degenerate input).
        detected_script: Script family detected in the *original* query.
        metadata: Step-level diagnostics collected during normalisation.
    """

    original: str
    normalized: str
    tokens: list[str]
    removed_stopwords: list[str] = field(default_factory=list)
    truncated: bool = False
    too_short: bool = False
    detected_script: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def token_count(self) -> int:
        """Number of tokens in the normalised query."""
        return len(self.tokens)

    @property
    def is_empty(self) -> bool:
        """``True`` when the normalised query contains no tokens."""
        return len(self.tokens) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe types only)."""
        return {
            "original": self.original,
            "normalized": self.normalized,
            "tokens": self.tokens,
            "removed_stopwords": self.removed_stopwords,
            "truncated": self.truncated,
            "too_short": self.too_short,
            "detected_script": self.detected_script,
            "token_count": self.token_count,
            "is_empty": self.is_empty,
        }


# ---------------------------------------------------------------------------
# Quoted-phrase extractor
# ---------------------------------------------------------------------------

_QUOTED_RE = re.compile(r'"([^"]*)"')


def _extract_quoted_phrases(text: str) -> tuple[str, list[str]]:
    """Replace ``"..."`` spans with placeholders and return extracted phrases.

    Returns:
        A tuple of *(placeholder_text, phrases)* where each phrase is the
        raw content between the double quotes.
    """
    phrases: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        idx = len(phrases)
        phrases.append(m.group(1))
        return f"__PHRASE_{idx}__"

    result = _QUOTED_RE.sub(_sub, text)
    return result, phrases


def _restore_quoted_phrases(tokens: list[str], phrases: list[str]) -> list[str]:
    """Expand ``__PHRASE_N__`` placeholders back to their original content."""
    out: list[str] = []
    placeholder_re = re.compile(r"^__phrase_(\d+)__$", re.IGNORECASE)
    for tok in tokens:
        m = placeholder_re.match(tok)
        if m:
            idx = int(m.group(1))
            if idx < len(phrases):
                # Preserve as a single quoted-phrase token (space-joined).
                out.append(phrases[idx].strip())
            else:
                out.append(tok)
        else:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Core normalizer
# ---------------------------------------------------------------------------


class QueryNormalizer:
    """Composable query preprocessing pipeline.

    Each normalisation step is applied sequentially according to
    :class:`NormalizerConfig`.  The result captures every transformation
    for downstream inspection or logging.

    Args:
        config: Normalisation configuration.  A default :class:`NormalizerConfig`
            (lowercase-only, no stopword removal) is used when not provided.

    Example::

        normalizer = QueryNormalizer(NormalizerConfig(remove_stopwords=True))
        result = normalizer.normalize("What are the best IR approaches?")
        print(result.normalized)   # "best ir approaches"
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        """Initialise the normalizer with an optional configuration.

        Args:
            config: Normalisation configuration.  Uses :class:`NormalizerConfig`
                defaults when ``None``.
        """
        self.config = config or NormalizerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, query: str) -> NormalizationResult:
        """Normalise *query* and return a :class:`NormalizationResult`.

        Args:
            query: Raw query string from the user.

        Returns:
            A :class:`NormalizationResult` with the processed query and
            step-level diagnostics.
        """
        cfg = self.config
        original = query
        metadata: dict[str, Any] = {}

        # --- Detect script before any transformation ---
        detected_script = detect_script(query)

        # --- Unicode normalisation ---
        if cfg.unicode_normalize:
            query = unicodedata.normalize("NFC", query)
            metadata["unicode_normalized"] = True

        # --- Quoted-phrase extraction ---
        phrases: list[str] = []
        if cfg.preserve_quoted_phrases:
            query, phrases = _extract_quoted_phrases(query)
            metadata["quoted_phrase_count"] = len(phrases)

        # --- Lowercase ---
        if cfg.lowercase:
            query = query.lower()

        # --- Contraction expansion ---
        if cfg.expand_contractions:
            query = _CONTRACTION_RE.sub(_replace_contraction, query)
            metadata["contractions_expanded"] = True

        # --- Punctuation ---
        query = self._apply_punctuation_policy(query, cfg.punctuation_policy)

        # --- Collapse whitespace ---
        if cfg.collapse_whitespace:
            query = re.sub(r"\s+", " ", query).strip()

        # --- Tokenise ---
        tokens = query.split()

        # --- Restore quoted phrases ---
        if cfg.preserve_quoted_phrases and phrases:
            tokens = _restore_quoted_phrases(tokens, phrases)

        # --- Stopword removal ---
        removed_stopwords: list[str] = []
        if cfg.remove_stopwords:
            stopwords = cfg.stopwords or frozenset()
            kept: list[str] = []
            for tok in tokens:
                # Never remove placeholder tokens or multi-word quoted phrases.
                if " " not in tok and tok in stopwords:
                    removed_stopwords.append(tok)
                else:
                    kept.append(tok)
            tokens = kept
            metadata["stopwords_removed"] = len(removed_stopwords)

        # --- Min token length ---
        if cfg.min_token_length > 1:
            before = len(tokens)
            tokens = [t for t in tokens if len(t) >= cfg.min_token_length]
            metadata["short_tokens_removed"] = before - len(tokens)

        # --- Extra steps ---
        for step_fn in cfg.extra_steps:
            tokens = step_fn(tokens)

        # --- Truncation ---
        truncated = False
        if cfg.max_tokens and len(tokens) > cfg.max_tokens:
            tokens = tokens[: cfg.max_tokens]
            truncated = True
            metadata["truncated_at"] = cfg.max_tokens

        # --- Assemble final string ---
        normalized = " ".join(tokens)

        # --- Short-query flag ---
        too_short = len(normalized) < cfg.min_query_length

        return NormalizationResult(
            original=original,
            normalized=normalized,
            tokens=tokens,
            removed_stopwords=removed_stopwords,
            truncated=truncated,
            too_short=too_short,
            detected_script=detected_script,
            metadata=metadata,
        )

    def batch_normalize(self, queries: list[str]) -> list[NormalizationResult]:
        """Normalise a list of queries.

        Args:
            queries: Raw query strings.

        Returns:
            A list of :class:`NormalizationResult` objects in the same order.
        """
        return [self.normalize(q) for q in queries]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_punctuation_policy(text: str, policy: PunctuationPolicy) -> str:
        if policy == PunctuationPolicy.REMOVE:
            return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        if policy == PunctuationPolicy.REPLACE_WITH_SPACE:
            return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        # KEEP — do nothing
        return text


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def make_default_normalizer() -> QueryNormalizer:
    """Return a normalizer suitable for general English text retrieval.

    Applies lowercase, Unicode NFC, contraction expansion, punctuation
    replacement, and whitespace collapsing.  Stopword removal is **off**
    by default so that phrase-match retrievers are not impacted.
    """
    return QueryNormalizer(
        NormalizerConfig(
            lowercase=True,
            unicode_normalize=True,
            expand_contractions=True,
            punctuation_policy=PunctuationPolicy.REPLACE_WITH_SPACE,
            collapse_whitespace=True,
            remove_stopwords=False,
        )
    )


def make_keyword_normalizer(stopwords: frozenset[str] | set[str] | None = None) -> QueryNormalizer:
    """Return a normalizer tuned for sparse / keyword retrieval.

    Stopword removal is enabled.  Single-character tokens are discarded.
    Quoted phrases are preserved so that exact-match operators still work.

    Args:
        stopwords: Custom stopword set.  Defaults to :data:`ENGLISH_STOPWORDS`.
    """
    return QueryNormalizer(
        NormalizerConfig(
            lowercase=True,
            unicode_normalize=True,
            expand_contractions=True,
            punctuation_policy=PunctuationPolicy.REPLACE_WITH_SPACE,
            collapse_whitespace=True,
            remove_stopwords=True,
            stopwords=stopwords,
            min_token_length=2,
            preserve_quoted_phrases=True,
        )
    )


def make_strict_normalizer(max_tokens: int = 32) -> QueryNormalizer:
    """Return a normalizer that enforces hard token-count limits.

    Useful for embedding models with tight input constraints.

    Args:
        max_tokens: Maximum number of tokens to retain.
    """
    return QueryNormalizer(
        NormalizerConfig(
            lowercase=True,
            unicode_normalize=True,
            expand_contractions=True,
            punctuation_policy=PunctuationPolicy.REPLACE_WITH_SPACE,
            collapse_whitespace=True,
            remove_stopwords=True,
            min_token_length=2,
            max_tokens=max_tokens,
        )
    )
