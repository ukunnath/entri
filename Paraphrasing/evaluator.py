import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict

warnings.filterwarnings("ignore")


try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    for _pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{_pkg}")
        except LookupError:
            nltk.download(_pkg, quiet=True)
    _BLEU_AVAILABLE = True
except ImportError:
    _BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    import numpy as _np
    _SIM_AVAILABLE = True
except ImportError:
    _SIM_AVAILABLE = False


@dataclass
class EvalResult:
    """Evaluation result for a single (source, paraphrase) pair."""

    source:     str
    paraphrase: str

    bleu:       Optional[float] = None   # 0–1, lower = more original surface
    rouge1:     Optional[float] = None
    rouge2:     Optional[float] = None
    rougeL:     Optional[float] = None
    similarity: Optional[float] = None  # 0–1, higher = more meaning preserved

    originality:  Optional[float] = None   # 1 − bleu
    quality_score: Optional[float] = None  # composite

    def summary(self) -> str:
        parts = [f"  Source     : {self.source[:70]}"]
        parts.append(f"  Paraphrase : {self.paraphrase[:70]}")
        if self.bleu      is not None: parts.append(f"  BLEU       : {self.bleu:.4f}")
        if self.rouge1    is not None: parts.append(f"  ROUGE-1    : {self.rouge1:.4f}")
        if self.rouge2    is not None: parts.append(f"  ROUGE-2    : {self.rouge2:.4f}")
        if self.rougeL    is not None: parts.append(f"  ROUGE-L    : {self.rougeL:.4f}")
        if self.similarity is not None: parts.append(f"  Similarity : {self.similarity:.4f}")
        if self.originality is not None: parts.append(f"  Originality: {self.originality:.4f}")
        if self.quality_score is not None:
            parts.append(f"  ─── Quality: {self.quality_score:.4f} / 1.00")
        return "\n".join(parts)


@dataclass
class EvalReport:
    """Aggregated evaluation report for multiple paraphrases."""

    results:  List[EvalResult] = field(default_factory=list)
    averages: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 60, "  EVALUATION REPORT", "=" * 60]
        for i, r in enumerate(self.results, 1):
            lines.append(f"\n  Candidate #{i}")
            lines.append(r.summary())
        if self.averages:
            lines.append("\n" + "─" * 60)
            lines.append("  AVERAGES")
            for k, v in self.averages.items():
                lines.append(f"    {k:<14}: {v:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)



class Evaluator:

    def __init__(
        self,
        sim_model: str = "all-MiniLM-L6-v2",
        verbose: bool  = True,
    ):
        self.verbose = verbose
        self._scorer = None
        self._sim    = None

        if _ROUGE_AVAILABLE:
            self._scorer = _rouge_scorer_mod.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

        if _SIM_AVAILABLE:
            try:
                if verbose:
                    print(f"[Evaluator] Loading similarity model '{sim_model}' …")
                self._sim = _SentenceTransformer(sim_model)
                if verbose:
                    print("[Evaluator] Similarity model ready.\n")
            except Exception as exc:
                if verbose:
                    print(f"[Evaluator] Similarity model failed ({exc}).")

        if verbose:
            avail = []
            if _BLEU_AVAILABLE: avail.append("BLEU")
            if _ROUGE_AVAILABLE: avail.append("ROUGE")
            if _SIM_AVAILABLE and self._sim: avail.append("Semantic Similarity")
            print(f"[Evaluator] Active metrics: {', '.join(avail) or 'none'}")



    def evaluate(self, source: str, paraphrases: List[str]) -> EvalReport:
        report = EvalReport()

        # Pre-compute embeddings in batch for efficiency
        embeddings = {}
        if self._sim is not None:
            all_texts   = [source] + paraphrases
            all_embeds  = self._sim.encode(all_texts, convert_to_numpy=True)
            src_embed   = all_embeds[0]
            para_embeds = all_embeds[1:]
            embeddings  = {i: para_embeds[i] for i in range(len(paraphrases))}
        else:
            src_embed = None

        for i, para in enumerate(paraphrases):
            r = EvalResult(source=source, paraphrase=para)

            # BLEU
            if _BLEU_AVAILABLE:
                ref     = source.lower().split()
                hyp     = para.lower().split()
                smooth  = SmoothingFunction().method1
                r.bleu  = round(sentence_bleu([ref], hyp, smoothing_function=smooth), 4)
                r.originality = round(1.0 - r.bleu, 4)

            # ROUGE
            if self._scorer is not None:
                scores  = self._scorer.score(source, para)
                r.rouge1 = round(scores["rouge1"].fmeasure, 4)
                r.rouge2 = round(scores["rouge2"].fmeasure, 4)
                r.rougeL = round(scores["rougeL"].fmeasure, 4)

            # Semantic similarity
            if src_embed is not None and i in embeddings:
                r.similarity = round(
                    float(_np.dot(src_embed, embeddings[i]) /
                          (_np.linalg.norm(src_embed) * _np.linalg.norm(embeddings[i]) + 1e-8)),
                    4
                )

            parts = []
            if r.similarity is not None: parts.append(("sim", r.similarity, 0.6))
            if r.originality is not None: parts.append(("ori", r.originality, 0.4))
            if parts:
                total_w = sum(w for _, _, w in parts)
                r.quality_score = round(
                    sum(v * w for _, v, w in parts) / total_w, 4
                )

            report.results.append(r)

        # Compute averages
        metric_keys = ["bleu", "rouge1", "rouge2", "rougeL", "similarity",
                       "originality", "quality_score"]
        for key in metric_keys:
            values = [getattr(r, key) for r in report.results
                      if getattr(r, key) is not None]
            if values:
                report.averages[key] = round(sum(values) / len(values), 4)

        return report


    @staticmethod
    def pick_best(report: EvalReport) -> Optional[EvalResult]:
        """Return the EvalResult with the highest quality_score."""
        candidates = [r for r in report.results if r.quality_score is not None]
        if not candidates:
            return report.results[0] if report.results else None
        return max(candidates, key=lambda r: r.quality_score)
