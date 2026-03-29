import re
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

try:
    from spellchecker import SpellChecker as _SpellChecker
    _SPELL_AVAILABLE = True
except ImportError:
    _SPELL_AVAILABLE = False

try:
    import language_tool_python as _ltp
    _LTP_AVAILABLE = True
except ImportError:
    _LTP_AVAILABLE = False

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    _FLUENCY_AVAILABLE = True
except ImportError:
    _FLUENCY_AVAILABLE = False



@dataclass
class QualityReport:
    """Holds all quality-check results for a single text string."""

    text: str

    # Spelling
    spelling_errors: List[str]      = field(default_factory=list)
    spelling_score: float           = 1.0   # 1.0 = perfect

    # Grammar
    grammar_issues: List[str]       = field(default_factory=list)
    grammar_score: float            = 1.0

    # Fluency
    perplexity: Optional[float]     = None
    fluency_score: float            = 1.0   # 1.0 = very fluent

    # Composite
    overall_score: float            = 1.0   # weighted average

    def summary(self) -> str:
        lines = [
            f"  Text            : {self.text[:80]}{'…' if len(self.text)>80 else ''}",
            f"  Spelling score  : {self.spelling_score:.2f}  "
            f"({'OK' if not self.spelling_errors else ', '.join(self.spelling_errors[:5])})",
            f"  Grammar score   : {self.grammar_score:.2f}  "
            f"({'OK' if not self.grammar_issues else self.grammar_issues[0]})",
            f"  Fluency score   : {self.fluency_score:.2f}"
            + (f"  (perplexity={self.perplexity:.1f})" if self.perplexity else ""),
            f"  ─── Overall     : {self.overall_score:.2f} / 1.00",
        ]
        return "\n".join(lines)



class QualityChecker:
    """
    Runs spelling, grammar and fluency checks on a piece of text and
    returns a QualityReport.

    Parameters
    ----------
    language : str
        BCP-47 language code used by LanguageTool (default: 'en-US').
    fluency_model : str
        Hugging Face model ID for the language model used to compute
        perplexity.  Must be a causal LM (GPT-style).
    verbose : bool
        Log loading progress.
    """

    # Perplexity thresholds — empirically determined on GPT-2
    _PPL_EXCELLENT = 40
    _PPL_GOOD      = 80
    _PPL_FAIR      = 160
    _PPL_POOR      = 320

    def __init__(
        self,
        language: str      = "en-US",
        fluency_model: str = "gpt2",
        verbose: bool      = True,
    ):
        self.language = language
        self.verbose  = verbose

        # ── Spell checker ──────────────────────────────────────────────
        if _SPELL_AVAILABLE:
            self._spell = _SpellChecker(language="en")
            if verbose:
                print("[QualityChecker] Spell checker loaded.")
        else:
            self._spell = None
            if verbose:
                print("[QualityChecker] pyspellchecker not found — spelling check disabled.")

        # ── LanguageTool grammar checker ───────────────────────────────
        if _LTP_AVAILABLE:
            try:
                self._lt = _ltp.LanguageTool(language)
                if verbose:
                    print("[QualityChecker] LanguageTool loaded.")
            except Exception as exc:
                self._lt = None
                if verbose:
                    print(f"[QualityChecker] LanguageTool unavailable ({exc}). "
                          "Using heuristic grammar checks.")
        else:
            self._lt = None
            if verbose:
                print("[QualityChecker] language_tool_python not found — "
                      "using heuristic grammar checks.")

        # ── GPT-2 for fluency (perplexity) ─────────────────────────────
        self._lm_model     = None
        self._lm_tokenizer = None
        if _FLUENCY_AVAILABLE:
            try:
                if verbose:
                    print(f"[QualityChecker] Loading fluency model '{fluency_model}' …")
                self._lm_tokenizer = GPT2TokenizerFast.from_pretrained(fluency_model)
                self._lm_model     = GPT2LMHeadModel.from_pretrained(fluency_model)
                self._lm_model.eval()
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._lm_model.to(self._device)
                if verbose:
                    print("[QualityChecker] Fluency model ready.\n")
            except Exception as exc:
                if verbose:
                    print(f"[QualityChecker] Fluency model failed ({exc}).")
        else:
            if verbose:
                print("[QualityChecker] transformers/torch not found — fluency check disabled.\n")


    # Public API


    def check(self, text: str) -> QualityReport:
        """
        Run all available quality checks on *text* and return a QualityReport.
        """
        report = QualityReport(text=text)

        report.spelling_errors, report.spelling_score = self._check_spelling(text)
        report.grammar_issues,  report.grammar_score  = self._check_grammar(text)
        report.perplexity,      report.fluency_score  = self._check_fluency(text)

        # Weighted composite score  (spelling 30%, grammar 40%, fluency 30%)
        report.overall_score = (
            0.30 * report.spelling_score +
            0.40 * report.grammar_score  +
            0.30 * report.fluency_score
        )
        return report

    # Private helpers


    def _check_spelling(self, text: str) -> Tuple[List[str], float]:
        """Return (misspelled_words, score∈[0,1])."""
        if self._spell is None:
            return [], 1.0

        # Tokenise into plain words (strip punctuation)
        words = re.findall(r"[a-zA-Z']+", text)
        if not words:
            return [], 1.0

        misspelled = list(self._spell.unknown(words))
        # Filter out proper nouns (capitalised words not at sentence start)
        filtered = [w for w in misspelled if w[0].islower()]

        score = max(0.0, 1.0 - len(filtered) / len(words))
        return filtered, round(score, 4)

    def _check_grammar(self, text: str) -> Tuple[List[str], float]:
        """Return (issue_messages, score∈[0,1])."""
        if self._lt is not None:
            return self._lt_grammar(text)
        return self._heuristic_grammar(text)

    def _lt_grammar(self, text: str) -> Tuple[List[str], float]:
        """LanguageTool-based grammar check."""
        matches = self._lt.check(text)
        issues  = [m.message for m in matches]
        words   = len(text.split())
        score   = max(0.0, 1.0 - len(matches) / max(words, 1))
        return issues, round(score, 4)

    @staticmethod
    def _heuristic_grammar(text: str) -> Tuple[List[str], float]:
        """
        Lightweight heuristic grammar checks when LanguageTool is absent.
        Checks: double spaces, missing capitalisation, double punctuation,
        sentence not ending with punctuation, common word repetitions.
        """
        issues: List[str] = []

        if "  " in text:
            issues.append("Double space detected.")
        if text and text[0].islower():
            issues.append("Sentence does not start with a capital letter.")
        if re.search(r"[.!?]{2,}", text):
            issues.append("Double punctuation detected.")
        if text and text[-1] not in ".!?":
            issues.append("Sentence does not end with punctuation.")

        # Detect immediately repeated words: "the the"
        repeated = re.findall(r"\b(\w+)\s+\1\b", text, re.IGNORECASE)
        if repeated:
            issues.append(f"Repeated word(s): {repeated}")

        score = max(0.0, 1.0 - len(issues) * 0.2)
        return issues, round(score, 4)

    def _check_fluency(self, text: str) -> Tuple[Optional[float], float]:
        """
        Compute GPT-2 perplexity as a proxy for fluency.
        Returns (perplexity, fluency_score∈[0,1]).
        Lower perplexity → more fluent.
        """
        if self._lm_model is None or self._lm_tokenizer is None:
            return None, 1.0

        try:
            encodings = self._lm_tokenizer(text, return_tensors="pt").to(self._device)
            input_ids = encodings.input_ids

            # Skip if text is too short for a meaningful perplexity
            if input_ids.shape[1] < 3:
                return None, 1.0

            with torch.no_grad():
                outputs = self._lm_model(input_ids, labels=input_ids)
                loss    = outputs.loss.item()

            perplexity = math.exp(loss)

            # Map perplexity to a [0,1] fluency score
            if perplexity <= self._PPL_EXCELLENT:
                fluency = 1.0
            elif perplexity <= self._PPL_GOOD:
                fluency = 0.85
            elif perplexity <= self._PPL_FAIR:
                fluency = 0.65
            elif perplexity <= self._PPL_POOR:
                fluency = 0.40
            else:
                fluency = 0.20

            return round(perplexity, 2), round(fluency, 4)

        except Exception:
            return None, 1.0
