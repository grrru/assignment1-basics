from collections import Counter

import pickle
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

    # pre_tokens을 tuple이 아닌 list로 두어 merge 시 바로 수정 가능하도록 함.
    pre_tokens: list[list[bytes]] = []
    pre_token_count: list[int] = []

    for k, v in str_pre_tokens.items():
        bk = k.encode("utf-8")
        li = list(bk[i : i + 1] for i in range(len(bk)))
        pre_tokens.append(li)
        pre_token_count.append(v)

    # 2. merge pair
    merges: list[tuple[bytes, bytes]] = []
    pairs: dict[tuple[bytes, bytes], int] = {}
    pre_token_id_for_pairs: dict[tuple[bytes, bytes], set[int]] = {}

    # 2-1. init pairs
    for i in range(len(pre_tokens)):
        pre_token = pre_tokens[i]
        cnt = pre_token_count[i]

        for j in range(len(pre_token) - 1):
            p: tuple[bytes, bytes] = (pre_token[j], pre_token[j + 1])
            pairs[p] = pairs.get(p, 0) + cnt
            if p not in pre_token_id_for_pairs:
                pre_token_id_for_pairs[p] = set()
            pre_token_id_for_pairs[p].add(i)

    # 2-2. merge and update
    while len(vocab) < vocab_size:
        if not pairs:
            break
        # find max count pair (count first, byte order second)
        max_pair = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(max_pair)
        max_pair_bytes = b"".join(max_pair)
        vocab[len(vocab)] = max_pair_bytes
        pairs.pop(max_pair)

        indices: set[int] = pre_token_id_for_pairs.pop(max_pair)
        for idx in indices:
            new_list: list[bytes] = []
            t = 0
            while t < len(pre_tokens[idx]):
                if t != len(pre_tokens[idx]) - 1 and (pre_tokens[idx][t], pre_tokens[idx][t + 1]) == max_pair:
                    new_list.append(max_pair_bytes)
                    t += 2
                else:
                    new_list.append(pre_tokens[idx][t])
                    t += 1

            prev_d: dict[tuple[bytes, bytes], int] = {}
            new_d: dict[tuple[bytes, bytes], int] = {}
            pair_set: set[tuple[bytes, bytes]] = set()

            for i in range(len(pre_tokens[idx]) - 1):
                p: tuple[bytes, bytes] = (pre_tokens[idx][i], pre_tokens[idx][i + 1])
                if p == max_pair:
                    continue
                prev_d[p] = prev_d.get(p, 0) + 1
                pair_set.add(p)

            for i in range(len(new_list) - 1):
                p: tuple[bytes, bytes] = (new_list[i], new_list[i + 1])
                new_d[p] = new_d.get(p, 0) + 1
                pair_set.add(p)

            for p in pair_set:
                diff = new_d.get(p, 0) - prev_d.get(p, 0)
                pairs[p] = pairs.get(p, 0) + diff * pre_token_count[idx]
                if pairs[p] == 0:
                    pairs.pop(p)

                if prev_d.get(p, 0) == 0:
                    if p not in pre_token_id_for_pairs:
                        pre_token_id_for_pairs[p] = set()
                    pre_token_id_for_pairs[p].add(idx)

                if new_d.get(p, 0) == 0:
                    if p in pre_token_id_for_pairs:
                        pre_token_id_for_pairs[p].discard(idx)

            pre_tokens[idx] = new_list

    return vocab, merges


# train tinystories
def train_bpe_tinystories():
    vocab, merges = train_bpe(
        "/Users/grrru/workspace/stanford-cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
    )

    pickle_tokenizer(vocab, merges, "TinyStories")


def pickle_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], prefix: str):
    with open(f"{prefix}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(f"{prefix}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    train_bpe_tinystories()
