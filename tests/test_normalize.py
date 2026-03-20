"""Tests for lexisearch.retrieval.normalize — QueryNormalizer pipeline."""

from __future__ import annotations

from lexisearch.retrieval.normalize import (
    ENGLISH_STOPWORDS,
    NormalizationResult,
    NormalizerConfig,
    PunctuationPolicy,
    QueryNormalizer,
    detect_script,
    make_default_normalizer,
    make_keyword_normalizer,
    make_strict_normalizer,
)

# ---------------------------------------------------------------------------
# detect_script
# ---------------------------------------------------------------------------


class TestDetectScript:
    def test_latin_english(self):
        assert detect_script("hello world") == "latin"

    def test_cjk_chinese(self):
        assert detect_script("你好世界") == "cjk"

    def test_cjk_japanese(self):
        assert detect_script("こんにちは") == "cjk"

    def test_arabic(self):
        assert detect_script("مرحبا بالعالم") == "arabic"

    def test_cyrillic(self):
        assert detect_script("Привет мир") == "cyrillic"

    def test_empty_string(self):
        assert detect_script("") == "unknown"

    def test_digits_only(self):
        # digits are not in any tracked set
        assert detect_script("12345") == "unknown"

    def test_mixed_prefers_dominant(self):
        # Far more latin characters → latin wins
        result = detect_script("Hello world — привет")
        assert result == "latin"


# ---------------------------------------------------------------------------
# NormalizerConfig defaults
# ---------------------------------------------------------------------------


class TestNormalizerConfigDefaults:
    def test_stopwords_default_to_english(self):
        cfg = NormalizerConfig()
        assert cfg.stopwords is not None
        assert "the" in cfg.stopwords

    def test_custom_stopwords_preserved(self):
        custom = {"foo", "bar"}
        cfg = NormalizerConfig(stopwords=custom)
        assert cfg.stopwords == custom

    def test_punctuation_policy_default(self):
        cfg = NormalizerConfig()
        assert cfg.punctuation_policy == PunctuationPolicy.REPLACE_WITH_SPACE


# ---------------------------------------------------------------------------
# NormalizationResult helpers
# ---------------------------------------------------------------------------


class TestNormalizationResult:
    def _make(self, tokens: list[str], normalized: str = "") -> NormalizationResult:
        return NormalizationResult(
            original="raw",
            normalized=normalized or " ".join(tokens),
            tokens=tokens,
        )

    def test_token_count(self):
        r = self._make(["hello", "world"])
        assert r.token_count == 2

    def test_is_empty_true(self):
        r = self._make([])
        assert r.is_empty is True

    def test_is_empty_false(self):
        r = self._make(["x"])
        assert r.is_empty is False

    def test_to_dict_keys(self):
        r = self._make(["hello"])
        d = r.to_dict()
        for key in (
            "original",
            "normalized",
            "tokens",
            "removed_stopwords",
            "truncated",
            "too_short",
            "detected_script",
            "token_count",
            "is_empty",
        ):
            assert key in d

    def test_to_dict_values(self):
        r = self._make(["hello", "world"])
        d = r.to_dict()
        assert d["token_count"] == 2
        assert d["is_empty"] is False


# ---------------------------------------------------------------------------
# QueryNormalizer — basic transformations
# ---------------------------------------------------------------------------


class TestQueryNormalizerLowercase:
    def test_converts_to_lowercase(self):
        n = QueryNormalizer(NormalizerConfig(lowercase=True))
        result = n.normalize("Hello WORLD")
        assert result.normalized == "hello world"

    def test_lowercase_off(self):
        n = QueryNormalizer(NormalizerConfig(lowercase=False, expand_contractions=False))
        result = n.normalize("Hello WORLD")
        assert "WORLD" in result.normalized


class TestQueryNormalizerUnicode:
    def test_nfc_normalisation(self):
        # "e" + combining acute → NFC: "é"
        composed = "e\u0301"
        n = QueryNormalizer(NormalizerConfig(unicode_normalize=True, expand_contractions=False))
        result = n.normalize(composed)
        # NFC form is a single code-point "\xe9"
        assert result.normalized == "\xe9"

    def test_unicode_off_leaves_decomposed(self):
        composed = "e\u0301"
        n = QueryNormalizer(
            NormalizerConfig(
                unicode_normalize=False,
                expand_contractions=False,
                lowercase=False,
                punctuation_policy=PunctuationPolicy.KEEP,
            )
        )
        result = n.normalize(composed)
        assert "\u0301" in result.normalized


class TestQueryNormalizerContractions:
    def test_dont_expanded(self):
        n = QueryNormalizer(NormalizerConfig(expand_contractions=True))
        result = n.normalize("I don't know")
        assert "do not" in result.normalized

    def test_wont_expanded(self):
        n = QueryNormalizer(NormalizerConfig(expand_contractions=True))
        result = n.normalize("It won't work")
        assert "will not" in result.normalized

    def test_its_expanded(self):
        n = QueryNormalizer(NormalizerConfig(expand_contractions=True))
        result = n.normalize("It's fine")
        assert "it is" in result.normalized

    def test_contractions_off(self):
        # With expand_contractions=False the apostrophe is handled by the
        # punctuation policy (default: REPLACE_WITH_SPACE), so "don't" may
        # become "don t" — what matters is that the expansion did NOT fire.
        n = QueryNormalizer(NormalizerConfig(expand_contractions=False))
        result = n.normalize("I don't know")
        assert "do not" not in result.normalized


class TestQueryNormalizerPunctuation:
    def test_remove_punctuation(self):
        n = QueryNormalizer(
            NormalizerConfig(
                punctuation_policy=PunctuationPolicy.REMOVE,
                expand_contractions=False,
                lowercase=False,
            )
        )
        result = n.normalize("Hello, world! How are you?")
        assert "," not in result.normalized
        assert "!" not in result.normalized
        assert "?" not in result.normalized

    def test_replace_with_space(self):
        n = QueryNormalizer(
            NormalizerConfig(
                punctuation_policy=PunctuationPolicy.REPLACE_WITH_SPACE,
                expand_contractions=False,
                lowercase=False,
                collapse_whitespace=True,
            )
        )
        result = n.normalize("Hello,world")
        assert "hello,world" not in result.normalized.lower()
        assert "hello" in result.normalized.lower()
        assert "world" in result.normalized.lower()

    def test_keep_punctuation(self):
        n = QueryNormalizer(
            NormalizerConfig(
                punctuation_policy=PunctuationPolicy.KEEP,
                expand_contractions=False,
                lowercase=False,
            )
        )
        result = n.normalize("Hello, world!")
        assert "," in result.normalized


class TestQueryNormalizerWhitespace:
    def test_collapses_multiple_spaces(self):
        n = QueryNormalizer(
            NormalizerConfig(
                collapse_whitespace=True,
                expand_contractions=False,
            )
        )
        result = n.normalize("hello   world  foo")
        assert "  " not in result.normalized

    def test_strips_leading_trailing(self):
        n = QueryNormalizer(
            NormalizerConfig(
                collapse_whitespace=True,
                expand_contractions=False,
            )
        )
        result = n.normalize("  hello world  ")
        assert result.normalized == result.normalized.strip()


# ---------------------------------------------------------------------------
# Stopword removal
# ---------------------------------------------------------------------------


class TestStopwordRemoval:
    def test_removes_stopwords(self):
        n = QueryNormalizer(
            NormalizerConfig(
                remove_stopwords=True,
                stopwords={"the", "is", "a"},
                expand_contractions=False,
            )
        )
        result = n.normalize("the cat is a mammal")
        assert "the" not in result.tokens
        assert "is" not in result.tokens
        assert "a" not in result.tokens
        assert "cat" in result.tokens
        assert "mammal" in result.tokens

    def test_removed_stopwords_tracked(self):
        n = QueryNormalizer(
            NormalizerConfig(
                remove_stopwords=True,
                stopwords={"the", "is"},
                expand_contractions=False,
            )
        )
        result = n.normalize("the cat is big")
        assert "the" in result.removed_stopwords
        assert "is" in result.removed_stopwords

    def test_stopword_removal_off(self):
        n = QueryNormalizer(
            NormalizerConfig(
                remove_stopwords=False,
                stopwords={"the"},
                expand_contractions=False,
            )
        )
        result = n.normalize("the quick brown fox")
        assert "the" in result.tokens

    def test_uses_english_stopwords_by_default(self):
        n = QueryNormalizer(NormalizerConfig(remove_stopwords=True))
        result = n.normalize("what are the best approaches")
        assert "the" not in result.tokens
        assert "best" in result.tokens
        assert "approaches" in result.tokens


# ---------------------------------------------------------------------------
# Min token length
# ---------------------------------------------------------------------------


class TestMinTokenLength:
    def test_drops_short_tokens(self):
        n = QueryNormalizer(
            NormalizerConfig(
                min_token_length=3,
                remove_stopwords=False,
                expand_contractions=False,
                punctuation_policy=PunctuationPolicy.REMOVE,
            )
        )
        result = n.normalize("I go to a big house")
        for tok in result.tokens:
            assert len(tok) >= 3, f"Short token found: {tok!r}"

    def test_keeps_tokens_at_threshold(self):
        n = QueryNormalizer(
            NormalizerConfig(
                min_token_length=2,
                remove_stopwords=False,
                expand_contractions=False,
            )
        )
        result = n.normalize("go to")
        assert "go" in result.tokens
        assert "to" in result.tokens


# ---------------------------------------------------------------------------
# Max tokens (truncation)
# ---------------------------------------------------------------------------


class TestMaxTokens:
    def test_truncates_to_max(self):
        n = QueryNormalizer(
            NormalizerConfig(
                max_tokens=3,
                remove_stopwords=False,
                expand_contractions=False,
            )
        )
        result = n.normalize("one two three four five")
        assert result.token_count == 3
        assert result.truncated is True

    def test_no_truncation_when_within_limit(self):
        n = QueryNormalizer(
            NormalizerConfig(
                max_tokens=10,
                remove_stopwords=False,
                expand_contractions=False,
            )
        )
        result = n.normalize("one two three")
        assert result.truncated is False

    def test_zero_max_tokens_means_no_limit(self):
        n = QueryNormalizer(
            NormalizerConfig(
                max_tokens=0,
                remove_stopwords=False,
                expand_contractions=False,
            )
        )
        long_query = " ".join([f"word{i}" for i in range(100)])
        result = n.normalize(long_query)
        assert result.token_count == 100
        assert result.truncated is False


# ---------------------------------------------------------------------------
# Too-short detection
# ---------------------------------------------------------------------------


class TestTooShort:
    def test_empty_query_is_too_short(self):
        n = QueryNormalizer()
        result = n.normalize("")
        assert result.too_short is True

    def test_single_char_too_short_with_default(self):
        n = QueryNormalizer(NormalizerConfig(min_query_length=2))
        result = n.normalize("x")
        assert result.too_short is True

    def test_normal_query_not_too_short(self):
        n = QueryNormalizer()
        result = n.normalize("information retrieval")
        assert result.too_short is False


# ---------------------------------------------------------------------------
# Quoted-phrase preservation
# ---------------------------------------------------------------------------


class TestQuotedPhrasePreservation:
    def test_quoted_phrase_kept_as_unit(self):
        n = QueryNormalizer(
            NormalizerConfig(
                preserve_quoted_phrases=True,
                remove_stopwords=True,
                expand_contractions=False,
            )
        )
        result = n.normalize('"machine learning" applications')
        # The quoted phrase should appear as one token in the result
        assert any("machine learning" in tok for tok in result.tokens)

    def test_stopwords_not_removed_from_phrase(self):
        n = QueryNormalizer(
            NormalizerConfig(
                preserve_quoted_phrases=True,
                remove_stopwords=True,
                stopwords={"the", "of"},
                expand_contractions=False,
            )
        )
        result = n.normalize('"state of the art" model')
        joined = " ".join(result.tokens)
        assert "state of the art" in joined

    def test_multiple_quoted_phrases(self):
        n = QueryNormalizer(
            NormalizerConfig(
                preserve_quoted_phrases=True,
                remove_stopwords=False,
                expand_contractions=False,
            )
        )
        result = n.normalize('"natural language" and "information retrieval"')
        joined = " ".join(result.tokens)
        assert "natural language" in joined
        assert "information retrieval" in joined

    def test_preserve_off_treats_quotes_normally(self):
        n = QueryNormalizer(
            NormalizerConfig(
                preserve_quoted_phrases=False,
                remove_stopwords=False,
                expand_contractions=False,
                punctuation_policy=PunctuationPolicy.REMOVE,
            )
        )
        result = n.normalize('"machine learning"')
        # Quotes stripped, words split
        assert "machine" in result.tokens
        assert "learning" in result.tokens


# ---------------------------------------------------------------------------
# Extra steps (plugin API)
# ---------------------------------------------------------------------------


class TestExtraSteps:
    def test_extra_step_applied(self):
        def uppercase_all(tokens: list[str]) -> list[str]:
            return [t.upper() for t in tokens]

        n = QueryNormalizer(
            NormalizerConfig(
                lowercase=False,
                expand_contractions=False,
                extra_steps=[uppercase_all],
            )
        )
        result = n.normalize("hello world")
        assert all(t == t.upper() for t in result.tokens)

    def test_multiple_extra_steps_ordered(self):
        log: list[int] = []

        def step_a(tokens: list[str]) -> list[str]:
            log.append(1)
            return tokens

        def step_b(tokens: list[str]) -> list[str]:
            log.append(2)
            return tokens

        n = QueryNormalizer(
            NormalizerConfig(
                expand_contractions=False,
                extra_steps=[step_a, step_b],
            )
        )
        n.normalize("test")
        assert log == [1, 2]

    def test_extra_step_can_filter(self):
        def drop_long(tokens: list[str]) -> list[str]:
            return [t for t in tokens if len(t) <= 5]

        n = QueryNormalizer(
            NormalizerConfig(
                expand_contractions=False,
                extra_steps=[drop_long],
            )
        )
        result = n.normalize("hi hello superlongword")
        assert "superlongword" not in result.tokens


# ---------------------------------------------------------------------------
# Script detection stored in result
# ---------------------------------------------------------------------------


class TestResultScriptDetection:
    def test_latin_detected(self):
        n = QueryNormalizer()
        result = n.normalize("search engine results")
        assert result.detected_script == "latin"

    def test_cjk_detected(self):
        n = QueryNormalizer()
        result = n.normalize("中文搜索引擎")
        assert result.detected_script == "cjk"


# ---------------------------------------------------------------------------
# batch_normalize
# ---------------------------------------------------------------------------


class TestBatchNormalize:
    def test_returns_same_count(self):
        n = QueryNormalizer()
        queries = ["hello world", "foo bar", "test"]
        results = n.batch_normalize(queries)
        assert len(results) == len(queries)

    def test_each_item_is_result(self):
        n = QueryNormalizer()
        results = n.batch_normalize(["a", "b"])
        for r in results:
            assert isinstance(r, NormalizationResult)

    def test_empty_list(self):
        n = QueryNormalizer()
        assert n.batch_normalize([]) == []

    def test_independent_processing(self):
        n = QueryNormalizer(NormalizerConfig(remove_stopwords=True))
        results = n.batch_normalize(["the cat", "a dog"])
        # "the" and "a" should be removed
        for r in results:
            assert "the" not in r.tokens
            assert "a" not in r.tokens


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_default_normalizer_lowercase(self):
        n = make_default_normalizer()
        result = n.normalize("HELLO WORLD")
        assert result.normalized == "hello world"

    def test_default_normalizer_no_stopword_removal(self):
        n = make_default_normalizer()
        result = n.normalize("what is the answer")
        # stopword removal off → "the" stays
        assert "the" in result.tokens

    def test_keyword_normalizer_removes_stopwords(self):
        n = make_keyword_normalizer()
        result = n.normalize("what is the best algorithm")
        assert "what" not in result.tokens
        assert "is" not in result.tokens
        assert "the" not in result.tokens
        assert "best" in result.tokens
        assert "algorithm" in result.tokens

    def test_keyword_normalizer_custom_stopwords(self):
        custom = frozenset({"alpha", "beta"})
        n = make_keyword_normalizer(stopwords=custom)
        result = n.normalize("alpha beta gamma")
        assert "alpha" not in result.tokens
        assert "beta" not in result.tokens
        assert "gamma" in result.tokens

    def test_strict_normalizer_truncates(self):
        n = make_strict_normalizer(max_tokens=5)
        long_query = " ".join([f"term{i}" for i in range(20)])
        result = n.normalize(long_query)
        assert result.token_count <= 5
        assert result.truncated is True

    def test_strict_normalizer_default_limit(self):
        n = make_strict_normalizer()
        long_query = " ".join([f"word{i}" for i in range(100)])
        result = n.normalize(long_query)
        assert result.token_count <= 32


# ---------------------------------------------------------------------------
# ENGLISH_STOPWORDS set
# ---------------------------------------------------------------------------


class TestEnglishStopwords:
    def test_common_words_present(self):
        for word in ("the", "a", "an", "in", "of", "is", "are", "and", "or"):
            assert word in ENGLISH_STOPWORDS

    def test_content_words_absent(self):
        for word in ("retrieval", "algorithm", "document", "query", "index"):
            assert word not in ENGLISH_STOPWORDS

    def test_is_frozenset(self):
        assert isinstance(ENGLISH_STOPWORDS, frozenset)


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline_english(self):
        n = QueryNormalizer(
            NormalizerConfig(
                lowercase=True,
                unicode_normalize=True,
                expand_contractions=True,
                punctuation_policy=PunctuationPolicy.REPLACE_WITH_SPACE,
                collapse_whitespace=True,
                remove_stopwords=True,
                min_token_length=2,
                max_tokens=10,
            )
        )
        raw = "  What ARE the BEST approaches for I.R. systems?  "
        result = n.normalize(raw)
        assert result.original == raw
        assert result.detected_script == "latin"
        assert result.too_short is False
        assert result.truncated is False
        assert result.token_count <= 10
        # Verify lowercase
        for tok in result.tokens:
            assert tok == tok.lower()

    def test_pipeline_with_quoted_phrase_and_stopwords(self):
        n = QueryNormalizer(
            NormalizerConfig(
                lowercase=True,
                remove_stopwords=True,
                preserve_quoted_phrases=True,
                expand_contractions=False,
            )
        )
        result = n.normalize('"vector search" and "neural retrieval" are the best')
        joined = " ".join(result.tokens)
        assert "vector search" in joined
        assert "neural retrieval" in joined
        # "the" should be removed (not inside quotes)
        assert "the" not in result.tokens

    def test_contraction_then_stopword_removal(self):
        n = QueryNormalizer(
            NormalizerConfig(
                expand_contractions=True,
                remove_stopwords=True,
                stopwords={"do", "not", "i"},
            )
        )
        result = n.normalize("I don't understand")
        # "don't" → "do not"; then "do", "not", "i" removed
        assert "do" not in result.tokens
        assert "not" not in result.tokens
        assert "understand" in result.tokens

    def test_metadata_populated(self):
        n = QueryNormalizer(
            NormalizerConfig(
                unicode_normalize=True,
                expand_contractions=True,
                remove_stopwords=True,
            )
        )
        result = n.normalize("the best algorithm")
        assert "unicode_normalized" in result.metadata
        assert "stopwords_removed" in result.metadata

    def test_empty_query_graceful(self):
        n = QueryNormalizer()
        result = n.normalize("")
        assert result.is_empty is True
        assert result.too_short is True
        assert result.normalized == ""

    def test_whitespace_only_query(self):
        n = QueryNormalizer()
        result = n.normalize("     ")
        assert result.is_empty is True
