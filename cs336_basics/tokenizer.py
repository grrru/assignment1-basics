from collections.abc import Iterable, Iterator
import json

from cs336_basics.common import CHAR_TO_BYTE


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab: dict[int, bytes] = {}
        with open(file=vocab_filepath, encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)

        for k, id in raw_vocab.items():
            vocab[id] = b"".join([CHAR_TO_BYTE[s].to_bytes() for s in k])

        merges: list[tuple[bytes, bytes]] = []
        with open(file=merges_filepath, encoding="utf-8") as f:
            for line in f:
                raws: list[str] = line.rstrip().split(" ")
                if len(raws) != 2:
                    print(f"merge must be a pair: {len(raws)}")
                    continue

                merges.append(
                    (
                        b"".join([CHAR_TO_BYTE[s].to_bytes() for s in raws[0]]),
                        b"".join([CHAR_TO_BYTE[s].to_bytes() for s in raws[1]]),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Enocde an input text into a sequence of token IDs.
        """

        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files
        that we cannot directly load into memory.
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        pass
