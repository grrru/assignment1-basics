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

    vocab: dict[int, bytes] = {}
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

    pre_tokens: dict[tuple[bytes, ...], int] = {}
    for k, v in str_pre_tokens.items():
        bk = k.encode("utf-8")
        t = tuple(bk[i : i + 1] for i in range(len(bk)))
        pre_tokens[t] = v

    # 2. merge pair
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # find max count pair (count first, byte order second)
        checked_pair: dict[tuple[bytes, ...], int] = {}
        for k, v in pre_tokens.items():
            for i in range(len(k) - 1):
                pair = k[i : i + 2]
                checked_pair[pair] = checked_pair.get(pair, 0) + v

        if not checked_pair:
            break
        max_pair = max(checked_pair.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append((max_pair[0], max_pair[1]))

        merged_pre_tokens: dict[tuple[bytes, ...], int] = {}
        for k, v in pre_tokens.items():
            i = 0
            n_k: list[bytes] = []
            while i < len(k):
                pair = k[i : i + 2]
                if pair == max_pair:
                    n_k.append(b"".join(pair))
                    i += 2
                else:
                    n_k.append(k[i : i + 1][0])
                    i += 1

            merged_pre_tokens[tuple(n_k)] = v

        pre_tokens = merged_pre_tokens
        vocab[len(vocab)] = b"".join(max_pair)

    return vocab, merges


print(
    train_bpe(
        "/Users/user/workspace/stanford-cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt",
        10000,
        ["<|endoftext|>"],
    )
)
