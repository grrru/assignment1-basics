from collections import Counter

import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Return vocab and merges

    vocab: The tokenizer vocabulary, a mapping from int to bytes
    merges: A list of BPE merges
    """

    # 0. Initialize vocab
    if vocab_size < 256 + len(special_tokens):
        raise ValueError

    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        idx = len(vocab)
        vocab[idx] = special_token.encode("utf-8")

    # 1. pre-tokenization
    with open(input_path, "rb") as f:
        file = f.read()

    # special_tokens이 빈 경우 splitted가 한 글자씩 잘리는 걸 방지
    if special_tokens:
        special_token_pat = "|".join(regex.escape(special_token) for special_token in special_tokens)
        splitted = regex.split(special_token_pat, file.decode("utf-8"))
    else:
        splitted = [file.decode("utf-8")]

    str_pre_tokens = Counter(match.group() for split in splitted for match in regex.finditer(PAT, split))

    pre_tokens = {}
    for k, v in str_pre_tokens.items():
        bk = k.encode("utf-8")
        t = tuple(bk[i : i + 1] for i in range(len(bk)))
        pre_tokens[t] = v

    print(pre_tokens)


train_bpe(
    "/Users/user/workspace/stanford-cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt",
    10000,
    ["<|endoftext|>"],
)
