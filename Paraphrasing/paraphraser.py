"""
paraphraser.py
==============
Core paraphrasing engine powered by the T5 (Text-To-Text Transfer Transformer)
model from Hugging Face Transformers.

Model used : Vamsi/T5_Paraphrase_Paws  (fine-tuned T5-base for paraphrasing)
Fallback   : t5-base with a custom prompt if the primary model is unavailable.

The module exposes a single high-level class: `Paraphraser`.
"""

import re
import warnings
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")

PRIMARY_MODEL   = "Vamsi/T5_Paraphrase_Paws"
FALLBACK_MODEL  = "t5-base"

DEFAULT_NUM_BEAMS         = 5
DEFAULT_NUM_RETURN_SEQS   = 3
DEFAULT_MAX_LENGTH        = 256
DEFAULT_MIN_LENGTH        = 10
DEFAULT_TEMPERATURE       = 1.5   # > 1 encourages diversity
DEFAULT_REPETITION_PENALTY = 2.5
DEFAULT_NO_REPEAT_NGRAM   = 3


class Paraphraser:
    """
    Generates paraphrases for a given input text using a pre-trained
    seq2seq transformer model.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.  Defaults to the T5 paraphrase model.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    verbose : bool
        Print loading progress when True.
    """

    def __init__(
        self,
        model_name: str = PRIMARY_MODEL,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.verbose    = verbose

        #  Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if verbose:
            print(f"[Paraphraser] Loading model '{model_name}' on {self.device} …")

        #Load tokenizer + model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as exc:
            if verbose:
                print(f"[Paraphraser] Primary model failed ({exc}). "
                      f"Falling back to '{FALLBACK_MODEL}' …")
            self.model_name = FALLBACK_MODEL
            self.tokenizer  = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            self.model      = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL)

        self.model.to(self.device)
        self.model.eval()

        if verbose:
            print("[Paraphraser] Model ready.\n")


    def paraphrase(
        self,
        text: str,
        num_return_sequences: int = DEFAULT_NUM_RETURN_SEQS,
        num_beams: int            = DEFAULT_NUM_BEAMS,
        max_length: int           = DEFAULT_MAX_LENGTH,
        min_length: int           = DEFAULT_MIN_LENGTH,
        temperature: float        = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM,
    ) -> List[str]:
        """
        Paraphrase *text* and return a list of candidate paraphrases.

        Parameters
        ----------
        text                 : Input string to paraphrase.
        num_return_sequences : How many distinct paraphrases to generate.
        num_beams            : Beam-search width (≥ num_return_sequences).
        max_length           : Maximum token length of each output.
        min_length           : Minimum token length of each output.
        temperature          : Sampling temperature (higher → more diverse).
        repetition_penalty   : Penalises repeated n-grams in the output.
        no_repeat_ngram_size : Forbids repeating n-grams of this size.

        Returns
        -------
        List[str] of unique, cleaned paraphrase strings.
        """
        text = text.strip()
        if not text:
            raise ValueError("Input text must not be empty.")

        # Guarantee beam width covers the requested number of sequences
        num_beams = max(num_beams, num_return_sequences)

        # Format input with task prefix expected by T5 paraphrase models
        prompt = self._build_prompt(text)

        # Tokenise
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids      = encoding["input_ids"],
                attention_mask = encoding["attention_mask"],
                max_length           = max_length,
                min_length           = min_length,
                num_beams            = num_beams,
                num_return_sequences = num_return_sequences,
                temperature          = temperature,
                repetition_penalty   = repetition_penalty,
                no_repeat_ngram_size = no_repeat_ngram_size,
                early_stopping       = True,
                do_sample            = False,   # deterministic beam search
            )

        # Decode & clean
        paraphrases = []
        for out in outputs:
            decoded = self.tokenizer.decode(out, skip_special_tokens=True)
            cleaned = self._clean(decoded)
            # Deduplicate and skip outputs identical to input
            if cleaned and cleaned.lower() != text.lower() and cleaned not in paraphrases:
                paraphrases.append(cleaned)

        return paraphrases if paraphrases else [text]   # graceful fallback


    def _build_prompt(self, text: str) -> str:
        """
        Wrap the raw text in the task prefix used during fine-tuning.
        The Vamsi/T5_Paraphrase_Paws model was trained with this prefix.
        For the plain t5-base fallback we use a natural-language instruction.
        """
        if "paraphrase" in self.model_name.lower() or "paws" in self.model_name.lower():
            return f"paraphrase: {text} </s>"
        # Generic T5 instruction format
        return f"paraphrase the following sentence: {text}"

    @staticmethod
    def _clean(text: str) -> str:
        """
        Post-process a decoded string:
          - Strip leading/trailing whitespace.
          - Collapse multiple spaces.
          - Capitalise first character.
          - Ensure terminal punctuation.
        """
        text = text.strip()
        text = re.sub(r" {2,}", " ", text)
        if not text:
            return text
        # Capitalise first letter
        text = text[0].upper() + text[1:]
        # Add period if no terminal punctuation
        if text[-1] not in ".!?":
            text += "."
        return text
