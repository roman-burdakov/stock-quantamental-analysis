"""
NVDA Quantamental Engine — NLP Feature Extraction.

Computes TF-IDF cosine similarity between consecutive filings,
keyword-dictionary scores for 9 tracked themes, and a combined
narrative-drift feature matrix.  Outputs NLPFeatureRecord schema.

Optional enhancements (enabled via config flags):
- Sentence-transformer embedding cosine distances (use_embeddings)
- LDA / BERTopic topic modeling (use_topic_model)
- Change-point detection via ruptures (use_change_point_detection)

Reqs: 7.1–7.9
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    ComponentStatus,
    ComponentStatusEnum,
    EngineConfig,
    KEYWORD_DICTIONARIES,
    NLPFeatureRecord,
    TextSectionRecord,
)

logger = logging.getLogger(__name__)


class NLPFeatureExtractor:
    """Extract NLP features from filing narrative sections."""

    TRACKED_THEMES: list[str] = list(KEYWORD_DICTIONARIES.keys())
    KEYWORD_DICTIONARIES: dict[str, list[str]] = KEYWORD_DICTIONARIES

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_tfidf_similarity(
        self, texts: list[TextSectionRecord]
    ) -> pd.DataFrame:
        """Compute pairwise TF-IDF cosine similarity between consecutive
        filings for Risk Factors and MD&A sections.

        NOTE: A single TF-IDF vectorizer is fitted on the full corpus of
        texts per section to ensure consistent vocabulary and IDF weights
        across all pairwise comparisons. This makes similarity scores
        comparable across filing pairs.

        Returns a DataFrame with columns:
            filing_date, source_available_date, source_accession,
            section, similarity_score, prev_filing_date
        """
        target_sections = ("risk_factors", "mda")
        rows: list[dict] = []

        for section_name in target_sections:
            # Filter to this section and sort chronologically
            section_texts = sorted(
                [t for t in texts if t.section_name == section_name and t.text],
                key=lambda t: t.filing_date,
            )
            if len(section_texts) < 2:
                continue

            # Fit a single vectorizer on the full corpus for this section
            # so that IDF weights and vocabulary are consistent across pairs.
            corpus = [t.text for t in section_texts]
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)

            for i in range(1, len(section_texts)):
                prev = section_texts[i - 1]
                curr = section_texts[i]

                # Compute cosine similarity using pre-fitted vectors
                sim = cosine_similarity(tfidf_matrix[i - 1:i], tfidf_matrix[i:i + 1])
                score = float(sim[0, 0])

                rows.append(
                    {
                        "filing_date": curr.filing_date,
                        "source_available_date": curr.source_available_date,
                        "source_accession": curr.accession_number,
                        "section": section_name,
                        "similarity_score": round(score, 6),
                        "prev_filing_date": prev.filing_date,
                    }
                )

        return pd.DataFrame(rows)

    def compute_keyword_scores(self, text: str) -> dict[str, float]:
        """Compute normalised keyword frequency per theme.

        Returns dict mapping theme name → score (count / total_words)
        for all 9 tracked themes.
        """
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words) if words else 1  # avoid division by zero

        scores: dict[str, float] = {}
        for theme, keywords in self.KEYWORD_DICTIONARIES.items():
            count = 0
            for kw in keywords:
                # Use regex word-boundary matching for multi-word keywords
                pattern = re.compile(r"\b" + re.escape(kw.lower()) + r"\b")
                count += len(pattern.findall(text_lower))
            scores[theme] = round(count / total_words, 6)

        return scores

    # ------------------------------------------------------------------
    # Sentiment polarity (Req 3.4 — v2 Task 2.3)
    # ------------------------------------------------------------------

    # Loughran-McDonald-style financial sentiment lexicon. We embed a
    # high-signal subset rather than depending on textblob/vader which are
    # trained on social media / movie reviews and consistently misclassify
    # words like "decline" or "concern" as neutral and "high" as positive.
    # The full LM dictionary has ~10K entries; this distilled subset
    # captures the most common terms in 10-K/10-Q filings.

    _LM_POSITIVE: tuple[str, ...] = (
        "achieve", "achievement", "achieved", "accomplish", "accomplished",
        "advantage", "advantageous", "advance", "advances", "advancing",
        "beneficial", "benefit", "benefited", "benefits",
        "boost", "boosted", "breakthrough", "compelling",
        "confidence", "confident", "constructive", "delight", "delighted",
        "deserved", "desirable", "despite", "drives", "driving",
        "effective", "efficient", "enable", "enabled", "enables", "enabling",
        "encourage", "encouraged", "encouraging", "enhance", "enhanced",
        "enhancement", "enhances", "enhancing", "enjoy", "enjoyed",
        "enthusiasm", "exceed", "exceeded", "exceeding", "exceeds",
        "excellent", "excellence", "exceptional", "exceptionally",
        "expand", "expanded", "expanding", "expansion",
        "favorable", "favorably", "favored", "fortunate", "fortunately",
        "gain", "gained", "gaining", "gains", "good",
        "great", "greatly", "greatness", "growth",
        "highest", "ideal", "ideally", "improve", "improved", "improvement",
        "improvements", "improves", "improving", "innovate", "innovation",
        "innovative", "lead", "leader", "leadership", "leading", "leads",
        "outperform", "outperformed", "outperforming", "outperforms",
        "popular", "popularity", "popularize", "positive", "positively",
        "premier", "premium", "profitable", "profitably", "profits",
        "progress", "promising", "prosper", "prosperity", "prosperous",
        "rebound", "rebounded", "record", "recovered", "recovery",
        "regain", "regained", "rejuvenated", "reward", "rewarded",
        "satisfaction", "satisfied", "satisfy", "satisfying",
        "stability", "stable", "stabilize", "stabilized",
        "strength", "strengthen", "strengthened", "strengthening",
        "strengthens", "strong", "stronger", "strongest", "strongly",
        "succeed", "succeeded", "success", "successes", "successful",
        "successfully", "superior", "surpass", "surpassed", "surpasses",
        "surpassing", "transformational", "tremendous", "tremendously",
        "unprecedented", "valuable", "wealth", "winner", "winning",
    )

    _LM_NEGATIVE: tuple[str, ...] = (
        "abandoned", "abandonment", "abandoning", "adverse", "adversely",
        "adversity", "alleged", "anomaly", "anomalies",
        "challenge", "challenged", "challenges", "challenging",
        "claim", "claimed", "claims", "complaint", "complaints",
        "concern", "concerned", "concerns", "constrain", "constrained",
        "constraints", "contraction", "contracted", "contractions",
        "crisis", "critical", "criticism", "criticized", "damage", "damaged",
        "damages", "danger", "dangerous", "decline", "declined", "declines",
        "declining", "decrease", "decreased", "decreases", "decreasing",
        "deficient", "deficiency", "deficiencies", "deficit", "deficits",
        "delayed", "delays", "deplete", "depleted", "depletion",
        "deteriorate", "deteriorated", "deterioration", "difficult",
        "difficulties", "difficulty", "diminish", "diminished",
        "diminishing", "disappoint", "disappointed", "disappointing",
        "disappointment", "disclose", "disclosed", "discrepancy",
        "disrupt", "disrupted", "disrupting", "disruption", "disruptions",
        "disruptive", "doubt", "doubted", "doubtful", "downgrade",
        "downturn", "downturns", "drag", "dragged", "dropping",
        "erratic", "erode", "eroded", "erosion", "exacerbate", "exacerbated",
        "exposure", "fail", "failed", "failing", "fails", "failure",
        "failures", "fall", "fallen", "falls", "fault", "faulty", "fear",
        "fears", "force", "forced", "fraud", "fraudulent",
        "harm", "harmed", "harmful", "headwind", "headwinds",
        "hinder", "hindered", "hindering",
        "impair", "impaired", "impairment", "impairments",
        "imposed", "improper", "improperly", "inability", "inadequate",
        "inadequately", "indictment", "ineffective", "inefficiencies",
        "inefficient", "inferior", "infringe", "infringed", "infringement",
        "insufficient", "insufficiency", "interfere", "interference",
        "interrupted", "interruption", "interruptions",
        "lacking", "lawsuit", "lawsuits", "litigation", "loss", "losses",
        "lost", "low", "lower", "lowered", "lowering", "lowest",
        "miss", "missed", "missing", "negative", "negatively",
        "obstacle", "obstacles", "obsolete", "obsolescence", "obstruction",
        "penalize", "penalized", "penalty", "penalties", "poor", "poorly",
        "pressure", "pressures", "problem", "problematic", "problems",
        "recall", "recalled", "recession", "recessionary",
        "regret", "regretfully", "restate", "restated", "restatement",
        "risk", "risks", "risky", "ruin", "ruined", "scrutiny",
        "setback", "setbacks", "shortfall", "shortfalls", "shortage",
        "shortages", "slow", "slowdown", "slowdowns", "slowed", "slower",
        "slowing", "slumped", "stagnant", "stagnation", "stop", "stopped",
        "subpoena", "suffer", "suffered", "suffering",
        "sued", "suing", "suspended", "suspension",
        "terminate", "terminated", "termination", "threat", "threats",
        "tough", "trouble", "troubled", "troubles", "troubling",
        "uncertain", "uncertainty", "uncertainties", "unattractive",
        "uncompetitive", "undesirable", "undermined", "undermining",
        "unexpected", "unexpectedly", "unfavorable", "unfavorably",
        "unfortunate", "unfortunately", "unstable", "unsuccessful",
        "violate", "violated", "violation", "violations",
        "vulnerable", "vulnerability", "warn", "warned", "warning",
        "warnings", "weak", "weaken", "weakened", "weakening", "weaker",
        "weakness", "weaknesses", "worse", "worsen", "worsened",
        "worsening", "worst",
    )

    def compute_sentiment_polarity(self, text: str) -> float:
        """Compute normalized sentiment polarity in [-1, +1] using the
        Loughran-McDonald financial-domain lexicon (Req 3.4).

        Returns ``(pos - neg) / (pos + neg + 1)`` where pos/neg are
        word-boundary-anchored counts of the embedded LM positive and
        negative word lists. The +1 in the denominator is a smoother that
        keeps the score finite when both counts are zero.

        Why a financial-domain lexicon: TextBlob and VADER are trained on
        movie reviews and social media and consistently mislabel financial
        text. For example, "high" registers as positive in VADER but is
        neutral in financial filings ("high cost of capital"). LM is the
        academic standard for 10-K/10-Q sentiment analysis.
        """
        if not text:
            return 0.0
        lowered = text.lower()
        pos = 0
        neg = 0
        for word in self._LM_POSITIVE:
            pattern = re.compile(r"\b" + re.escape(word) + r"\b")
            pos += len(pattern.findall(lowered))
        for word in self._LM_NEGATIVE:
            pattern = re.compile(r"\b" + re.escape(word) + r"\b")
            neg += len(pattern.findall(lowered))
        if pos + neg == 0:
            return 0.0
        return round((pos - neg) / (pos + neg + 1), 6)

    def compute_sentiment_delta(
        self, sections: list[TextSectionRecord]
    ) -> pd.DataFrame:
        """Compute per-filing sentiment_polarity and sentiment_delta
        (current vs prior filing of the same section type).

        Returns a DataFrame ready for inclusion in the NLP feature
        matrix, with columns matching NLPFeatureRecord schema.
        """
        rows: list[dict] = []
        # Sort by section then date for deterministic prior-filing pairing
        by_section: dict[str, list[TextSectionRecord]] = {}
        for s in sections:
            by_section.setdefault(s.section_name, []).append(s)
        for section_name, recs in by_section.items():
            recs_sorted = sorted(recs, key=lambda r: r.filing_date)
            prev_polarity: float | None = None
            prev_filing_date: str | None = None
            for r in recs_sorted:
                polarity = self.compute_sentiment_polarity(r.text)
                # Polarity feature
                rows.append(
                    {
                        "filing_date": r.filing_date,
                        "source_available_date": r.source_available_date,
                        "source_accession": r.accession_number,
                        "section": section_name,
                        "feature_type": "sentiment",
                        "feature_name": "sentiment_polarity",
                        "value": polarity,
                        "prev_filing_date": prev_filing_date,
                    }
                )
                # Delta feature (only after first observation)
                if prev_polarity is not None:
                    rows.append(
                        {
                            "filing_date": r.filing_date,
                            "source_available_date": r.source_available_date,
                            "source_accession": r.accession_number,
                            "section": section_name,
                            "feature_type": "sentiment",
                            "feature_name": "sentiment_delta",
                            "value": round(polarity - prev_polarity, 6),
                            "prev_filing_date": prev_filing_date,
                        }
                    )
                prev_polarity = polarity
                prev_filing_date = r.filing_date
        return pd.DataFrame(rows)

    def compute_narrative_drift(
        self, sections: list[TextSectionRecord]
    ) -> pd.DataFrame:
        """Build the full NLP feature matrix combining TF-IDF similarity
        and keyword scores, plus optional enhancements.

        Output conforms to NLPFeatureRecord schema.  Saves result to
        ``data/processed/nvda_nlp_features.csv``.
        """
        records: list[dict] = []

        # --- TF-IDF similarity features ---
        tfidf_df = self.compute_tfidf_similarity(sections)
        for _, row in tfidf_df.iterrows():
            records.append(
                {
                    "filing_date": row["filing_date"],
                    "source_available_date": row["source_available_date"],
                    "source_accession": row["source_accession"],
                    "section": row["section"],
                    "feature_type": "tfidf_similarity",
                    "feature_name": "cosine_similarity",
                    "value": row["similarity_score"],
                    "prev_filing_date": row["prev_filing_date"],
                }
            )

        # --- Keyword score features ---
        for sec in sections:
            if not sec.text:
                continue
            kw_scores = self.compute_keyword_scores(sec.text)
            for theme, score in kw_scores.items():
                records.append(
                    {
                        "filing_date": sec.filing_date,
                        "source_available_date": sec.source_available_date,
                        "source_accession": sec.accession_number,
                        "section": sec.section_name,
                        "feature_type": "keyword_score",
                        "feature_name": theme,
                        "value": score,
                        "prev_filing_date": None,
                    }
                )

        # --- Sentiment features (Req 3.4 — v2 Task 2.3) ---
        # Compute Loughran-McDonald polarity and per-section delta vs
        # prior filing. This produces sentiment_polarity (level) and
        # sentiment_delta (change) for every section that has text.
        sent_df = self.compute_sentiment_delta(sections)
        if not sent_df.empty:
            for _, row in sent_df.iterrows():
                records.append(row.to_dict())

        # --- Optional: Embedding distances (17.1) ---
        if self.config.use_embeddings:
            logger.info("Computing sentence-transformer embedding distances …")
            emb_df = self.compute_embedding_distance(sections)
            if not emb_df.empty:
                for _, row in emb_df.iterrows():
                    records.append(row.to_dict())
                logger.info("  → %d embedding distance rows", len(emb_df))

        # --- Optional: Topic modeling (17.2) ---
        if self.config.use_topic_model:
            logger.info("Fitting topic model …")
            topic_df, emerging_fading = self.fit_topic_model(sections)
            if not topic_df.empty:
                for _, row in topic_df.iterrows():
                    records.append(row.to_dict())
                logger.info("  → %d topic distribution rows", len(topic_df))
            if emerging_fading:
                logger.info("  → %d emerging/fading topics identified", len(emerging_fading))

        # --- Optional: Change-point detection (17.3) ---
        if self.config.use_change_point_detection:
            logger.info("Running change-point detection on NLP time series …")
            # Apply to TF-IDF similarity series
            if not tfidf_df.empty:
                for section_name in tfidf_df["section"].unique():
                    sec_df = tfidf_df[tfidf_df["section"] == section_name].sort_values("filing_date")
                    if len(sec_df) >= 4:
                        sim_series = pd.Series(
                            sec_df["similarity_score"].values,
                            index=sec_df["filing_date"].values,
                        )
                        cp_dates = self.detect_change_points(sim_series)
                        for cp_date in cp_dates:
                            records.append({
                                "filing_date": cp_date,
                                "source_available_date": cp_date,
                                "source_accession": "",
                                "section": section_name,
                                "feature_type": "change_point",
                                "feature_name": "tfidf_similarity_break",
                                "value": 1.0,
                                "prev_filing_date": None,
                            })

            # Apply to keyword score time series (aggregate across themes)
            kw_records_df = pd.DataFrame([
                r for r in records if r.get("feature_type") == "keyword_score"
            ])
            if not kw_records_df.empty:
                for theme in kw_records_df["feature_name"].unique():
                    theme_df = kw_records_df[kw_records_df["feature_name"] == theme].sort_values("filing_date")
                    if len(theme_df) >= 4:
                        kw_series = pd.Series(
                            theme_df["value"].values,
                            index=theme_df["filing_date"].values,
                        )
                        cp_dates = self.detect_change_points(kw_series)
                        for cp_date in cp_dates:
                            records.append({
                                "filing_date": cp_date,
                                "source_available_date": cp_date,
                                "source_accession": "",
                                "section": "all",
                                "feature_type": "change_point",
                                "feature_name": f"{theme}_break",
                                "value": 1.0,
                                "prev_filing_date": None,
                            })

        df = pd.DataFrame(records)

        # Persist to processed directory
        out_dir = Path(self.config.processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "nvda_nlp_features.csv"
        df.to_csv(out_path, index=False)

        return df

    def assess_nlp_quality(
        self,
        sections: list[TextSectionRecord],
        config: EngineConfig | None = None,
    ) -> ComponentStatus:
        """Assess NLP feature quality based on text extraction quality.

        Delegates to :class:`FilingTextParser.check_extraction_quality` to
        determine whether the underlying text is adequate.  When extraction
        quality is poor (``diagnostic_only``), NLP features are also marked
        ``diagnostic_only``.

        Parameters
        ----------
        sections:
            All ``TextSectionRecord`` instances across filings.
        config:
            Optional engine config for threshold overrides.

        Returns
        -------
        ComponentStatus
        """
        from src.filing_text_parser import FilingTextParser

        cfg = config or self.config
        parser = FilingTextParser(cfg)
        extraction_status = parser.check_extraction_quality(sections, cfg)

        if extraction_status.status == ComponentStatusEnum.DIAGNOSTIC_ONLY:
            return ComponentStatus(
                component_name="nlp_signal",
                status=ComponentStatusEnum.DIAGNOSTIC_ONLY,
                reason=(
                    f"NLP features are diagnostic-only because text extraction "
                    f"quality is poor: {extraction_status.reason}"
                ),
                details=extraction_status.details,
            )

        if extraction_status.status == ComponentStatusEnum.UNAVAILABLE:
            return ComponentStatus(
                component_name="nlp_signal",
                status=ComponentStatusEnum.UNAVAILABLE,
                reason=f"NLP features unavailable: {extraction_status.reason}",
                details=extraction_status.details,
            )

        return ComponentStatus(
            component_name="nlp_signal",
            status=ComponentStatusEnum.USABLE,
            reason="NLP features are usable; text extraction quality is adequate.",
            details=extraction_status.details,
        )

    def extract_shift_snippets(
        self, pairs: list[tuple[TextSectionRecord, TextSectionRecord]], n: int = 5
    ) -> list[dict]:
        """Extract examples of text that changed significantly between
        consecutive filings.

        *pairs* is a list of ``(prev_section, curr_section)`` tuples for
        the same section across consecutive filings.

        Returns up to *n* dicts with keys:
            section, prev_filing_date, curr_filing_date,
            prev_accession, curr_accession, similarity_score,
            prev_snippet, curr_snippet
        """
        scored: list[dict] = []

        for prev, curr in pairs:
            score = self._pairwise_tfidf_cosine(prev.text, curr.text)
            scored.append(
                {
                    "section": curr.section_name,
                    "prev_filing_date": prev.filing_date,
                    "curr_filing_date": curr.filing_date,
                    "prev_accession": prev.accession_number,
                    "curr_accession": curr.accession_number,
                    "similarity_score": round(score, 6),
                    "prev_snippet": self._snippet(prev.text),
                    "curr_snippet": self._snippet(curr.text),
                }
            )

        # Sort by lowest similarity (biggest change) and return top n
        scored.sort(key=lambda d: d["similarity_score"])
        return scored[:n]

    # ------------------------------------------------------------------
    # Optional: Sentence-transformer embedding distances (17.1)
    # ------------------------------------------------------------------

    def compute_embedding_distance(
        self, texts: list[TextSectionRecord]
    ) -> pd.DataFrame:
        """Compute sentence-transformer embedding cosine distances between
        consecutive filings for Risk Factors and MD&A sections.

        Uses ``all-MiniLM-L6-v2`` (lightweight, ~80 MB).  Gracefully
        returns an empty DataFrame when ``sentence-transformers`` is not
        installed.

        Returns DataFrame with NLPFeatureRecord-compatible columns:
            filing_date, source_available_date, source_accession,
            section, feature_type, feature_name, value, prev_filing_date
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — skipping embedding distance. "
                "Install with: pip install sentence-transformers"
            )
            return pd.DataFrame()

        model = SentenceTransformer("all-MiniLM-L6-v2")
        target_sections = ("risk_factors", "mda")
        rows: list[dict] = []

        for section_name in target_sections:
            section_texts = sorted(
                [t for t in texts if t.section_name == section_name and t.text],
                key=lambda t: t.filing_date,
            )
            if len(section_texts) < 2:
                continue

            # Encode all texts for this section at once
            corpus = [t.text for t in section_texts]
            embeddings = model.encode(corpus, show_progress_bar=False)

            for i in range(1, len(section_texts)):
                prev = section_texts[i - 1]
                curr = section_texts[i]

                # Cosine distance = 1 - cosine_similarity
                sim = float(cosine_similarity(
                    embeddings[i - 1:i], embeddings[i:i + 1]
                )[0, 0])
                distance = round(1.0 - sim, 6)

                rows.append({
                    "filing_date": curr.filing_date,
                    "source_available_date": curr.source_available_date,
                    "source_accession": curr.accession_number,
                    "section": section_name,
                    "feature_type": "embedding_distance",
                    "feature_name": "cosine_distance",
                    "value": distance,
                    "prev_filing_date": prev.filing_date,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Optional: Topic modeling — LDA primary, BERTopic if available (17.2)
    # ------------------------------------------------------------------

    def fit_topic_model(
        self, texts: list[TextSectionRecord], n_topics: int = 5
    ) -> tuple[pd.DataFrame, list[dict]]:
        """Fit a topic model on filing texts and produce per-filing topic
        distributions plus an emerging/fading topic table.

        Primary implementation uses sklearn LatentDirichletAllocation.
        Falls back gracefully if sklearn is unavailable (unlikely).
        Optionally uses BERTopic when available.

        Returns
        -------
        tuple[pd.DataFrame, list[dict]]
            - DataFrame of per-filing topic distributions (NLPFeatureRecord-
              compatible rows with feature_type="topic_distribution").
            - List of dicts describing emerging/fading topics across filings.
        """
        use_bertopic = False
        try:
            from bertopic import BERTopic  # noqa: F401
            use_bertopic = True
        except ImportError:
            logger.info(
                "bertopic not installed — using sklearn LDA for topic modeling."
            )

        # Collect texts with metadata, sorted chronologically
        doc_records = sorted(
            [t for t in texts if t.text and len(t.text.strip()) > 50],
            key=lambda t: t.filing_date,
        )
        if len(doc_records) < 2:
            logger.warning("Fewer than 2 documents — skipping topic modeling.")
            return pd.DataFrame(), []

        corpus = [t.text for t in doc_records]

        if use_bertopic:
            return self._fit_bertopic(doc_records, corpus, n_topics)
        return self._fit_lda(doc_records, corpus, n_topics)

    def _fit_lda(
        self,
        doc_records: list[TextSectionRecord],
        corpus: list[str],
        n_topics: int,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """LDA-based topic modeling using sklearn."""
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, stop_words="english", max_features=1000,
        )
        try:
            dtm = vectorizer.fit_transform(corpus)
        except ValueError:
            logger.warning("CountVectorizer produced empty vocabulary — skipping LDA.")
            return pd.DataFrame(), []

        actual_topics = min(n_topics, dtm.shape[0], dtm.shape[1])
        if actual_topics < 2:
            logger.warning("Not enough features/documents for LDA.")
            return pd.DataFrame(), []

        lda = LatentDirichletAllocation(
            n_components=actual_topics, random_state=42, max_iter=20,
        )
        doc_topic_matrix = lda.fit_transform(dtm)  # shape: (n_docs, n_topics)

        feature_names = vectorizer.get_feature_names_out()

        # Build per-filing topic distribution rows
        rows: list[dict] = []
        for idx, rec in enumerate(doc_records):
            for topic_idx in range(actual_topics):
                top_words = self._top_topic_words(
                    lda.components_[topic_idx], feature_names, n=5
                )
                topic_label = f"topic_{topic_idx}_{top_words}"
                rows.append({
                    "filing_date": rec.filing_date,
                    "source_available_date": rec.source_available_date,
                    "source_accession": rec.accession_number,
                    "section": rec.section_name,
                    "feature_type": "topic_distribution",
                    "feature_name": topic_label,
                    "value": round(float(doc_topic_matrix[idx, topic_idx]), 6),
                    "prev_filing_date": None,
                })

        df = pd.DataFrame(rows)

        # Build emerging/fading topic table
        emerging_fading = self._compute_emerging_fading(
            doc_records, doc_topic_matrix, lda, feature_names, actual_topics,
        )

        return df, emerging_fading

    def _fit_bertopic(
        self,
        doc_records: list[TextSectionRecord],
        corpus: list[str],
        n_topics: int,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """BERTopic-based topic modeling (used when bertopic is installed)."""
        try:
            from bertopic import BERTopic

            topic_model = BERTopic(nr_topics=n_topics, verbose=False)
            topics, probs = topic_model.fit_transform(corpus)

            rows: list[dict] = []
            for idx, rec in enumerate(doc_records):
                topic_id = topics[idx]
                topic_info = topic_model.get_topic(topic_id)
                if topic_info:
                    label = "_".join([w for w, _ in topic_info[:3]])
                else:
                    label = f"topic_{topic_id}"

                prob = float(probs[idx]) if probs is not None and idx < len(probs) else 0.0
                rows.append({
                    "filing_date": rec.filing_date,
                    "source_available_date": rec.source_available_date,
                    "source_accession": rec.accession_number,
                    "section": rec.section_name,
                    "feature_type": "topic_distribution",
                    "feature_name": f"bertopic_{label}",
                    "value": round(prob, 6),
                    "prev_filing_date": None,
                })

            return pd.DataFrame(rows), []
        except Exception as exc:
            logger.warning("BERTopic failed (%s) — falling back to LDA.", exc)
            return self._fit_lda(doc_records, corpus, n_topics)

    @staticmethod
    def _top_topic_words(
        component: np.ndarray, feature_names: np.ndarray, n: int = 5
    ) -> str:
        """Return top-n words for a topic component as a joined string."""
        top_idx = component.argsort()[-n:][::-1]
        return "_".join(feature_names[top_idx])

    @staticmethod
    def _compute_emerging_fading(
        doc_records: list[TextSectionRecord],
        doc_topic_matrix: np.ndarray,
        lda,
        feature_names: np.ndarray,
        n_topics: int,
    ) -> list[dict]:
        """Identify topics that are emerging (increasing weight) or fading
        (decreasing weight) across filings."""
        if doc_topic_matrix.shape[0] < 2:
            return []

        results: list[dict] = []
        first_half = doc_topic_matrix[: len(doc_records) // 2].mean(axis=0)
        second_half = doc_topic_matrix[len(doc_records) // 2 :].mean(axis=0)

        for t in range(n_topics):
            top_words_arr = lda.components_[t].argsort()[-5:][::-1]
            top_words = ", ".join(feature_names[top_words_arr])
            delta = float(second_half[t] - first_half[t])
            direction = "emerging" if delta > 0.01 else ("fading" if delta < -0.01 else "stable")
            results.append({
                "topic_id": t,
                "top_words": top_words,
                "early_weight": round(float(first_half[t]), 4),
                "late_weight": round(float(second_half[t]), 4),
                "delta": round(delta, 4),
                "direction": direction,
            })

        return results

    # ------------------------------------------------------------------
    # Optional: Change-point detection with ruptures (17.3)
    # ------------------------------------------------------------------

    def detect_change_points(
        self, series: pd.Series, pen: float = 3.0
    ) -> list[str]:
        """Detect structural breaks in an NLP feature time series using
        the ``ruptures`` library (Pelt algorithm, RBF kernel).

        Parameters
        ----------
        series : pd.Series
            Time series indexed by date strings (YYYY-MM-DD).
        pen : float
            Penalty value for the Pelt algorithm (higher = fewer breaks).

        Returns
        -------
        list[str]
            List of date strings where change points were detected.
            Empty list if ``ruptures`` is not installed or series is too short.
        """
        try:
            import ruptures as rpt
        except ImportError:
            logger.warning(
                "ruptures not installed — skipping change-point detection. "
                "Install with: pip install ruptures"
            )
            return []

        # Need at least 4 data points for meaningful detection
        clean = series.dropna()
        if len(clean) < 4:
            logger.info("Series too short (%d points) for change-point detection.", len(clean))
            return []

        signal = clean.values.astype(float).reshape(-1, 1)

        algo = rpt.Pelt(model="rbf", min_size=2).fit(signal)
        try:
            breakpoints = algo.predict(pen=pen)
        except Exception as exc:
            logger.warning("Change-point detection failed: %s", exc)
            return []

        # breakpoints are 1-based indices; last element is always len(signal)
        dates = list(clean.index)
        cp_dates: list[str] = []
        for bp in breakpoints:
            if bp < len(dates):  # exclude the terminal breakpoint
                cp_dates.append(str(dates[bp]))

        return cp_dates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_tfidf_cosine(text_a: str, text_b: str) -> float:
        """Compute TF-IDF cosine similarity between two texts."""
        if not text_a.strip() or not text_b.strip():
            return 0.0

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(sim[0, 0])

    @staticmethod
    def _snippet(text: str, max_chars: int = 300) -> str:
        """Return the first *max_chars* characters of *text* as a snippet."""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + " …"
