# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP
from __future__ import annotations

import gzip
import html
from pathlib import Path

import ftfy
import regex as re
import torch


class Tokenizer:
    """
    Text tokenizer that converts natural language (strings) to a list of token
    IDs using the Byte-Pair Encoding (BPE) algorithm. This implementation is
    slightly refactored from Open AI CLIP (https://github.com/openai/clip) and
    uses the same vocabulary of approximately 49K tokens.
    """

    def __init__(self):
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        # Make a mapping between utf-8 bytes and corresponding unicode strings.
        self.byte_encoder = {b: chr(b) for b in bs}
        n = 0
        for b in range(2**8):
            if b not in self.byte_encoder:
                self.byte_encoder[b] = chr(2**8 + n)
                n += 1

        bpe_path = Path(__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def __call__(self, text: str | list[str]) -> list[torch.IntTensor]:
        """
        Returns the tokenized representation of given input string(s).

        Args:
            text: An input string or list of strings to tokenize.

        Returns:
            List of tensors containing tokens. These tensors would also include
            the boundary tokens (start/end of sentence).
        """

        text_list = [text] if isinstance(text, str) else text

        token_tensors = []

        for text in text_list:
            bpe_tokens = []

            # Preprocess text like CLIP:
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            text = re.sub(r"\s+", " ", text)
            text = text.strip().lower()

            for token in re.findall(self.pat, text):
                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                bpe_tokens.extend(
                    self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
                )

            # Add boundary tokens after encoding:
            sot_token = self.encoder["<|startoftext|>"]
            eot_token = self.encoder["<|endoftext|>"]
            bpe_tokens = [sot_token, *bpe_tokens, eot_token]
            token_tensors.append(torch.IntTensor(bpe_tokens))

        return token_tensors

    @staticmethod
    def get_pairs(word: str) -> set[str]:
        """
        Return set of symbol pairs in a word.
        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self.get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word
