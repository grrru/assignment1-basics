def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Return vocab and merges

    vocab: The tokenizer vocabulary, a mapping from int to bytes
    merges: A list of BPE merges
    """

    if vocab_size < 257:
        raise ValueError

    # 0. Initialize vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        idx = len(vocab)
        vocab[idx] = special_token.encode("UTF-8")

    # 1. pre-tokenization


train_bpe("f", 10000, ["<|endoftext|>"])
