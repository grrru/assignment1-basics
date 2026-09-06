import argparse
import pickle

from cs336_basics import bpe

if __name__ == "__main__":
    """
    ## TinyStories 학습 결과
    1. non-chunking
        init vocab: 0.026ms
        pre-tokenization: 216829.838ms
        elapsed time: 247645.444ms
    2. chunking & parallelize pre-tokenization with 4 chunks
        init vocab: 0.024ms
        pre-tokenization: 73145.086ms
        elapsed time: 104329.446ms
    3. chunking & parallelize pre-tokenization with 8 chunks
        init vocab: 0.022ms
        pre-tokenization: 35489.720ms
        merge: 28103.325ms
        elapsed time: 63593.067ms
    4. chunking & parallelize pre-tokenization with 16 chunks (12core CPU)
        init vocab: 0.023ms
        pre-tokenization: 33896.956ms
        merge: 26315.086ms
        elapsed time: 60212.065ms
    5. chunking & parallelize pre-tokenization with 32 chunks (12core CPU)
        init vocab: 0.023ms
        pre-tokenization: 30908.055ms
        merge: 25901.002ms
        elapsed time: 56809.080ms
    6. optimize searching max_pair using heapq
        special_tokens:['<|endoftext|>']
        num_chunks:16
        init vocab: 0.021ms
        pre-tokenization: 32743.606ms
        merge: 10397.472ms
        elapsed time: 43141.098ms



    (b) Profile your code. What part of the tokenizer training process takes the most time?
        pre_tokenization에서 가장 많은 시간을 사용한다.(poll은 worker가 기다린 시간)
        merge 단계에서 가장 빈번한 pair를 찾는 시간이 그 다음 대부분을 차지한다.
 
        ncalls  tottime  percall  cumtime  percall filename:lineno(function)
               19   73.646    3.876   73.646    3.876 {method 'poll' of 'select.poll' objects}
            15326   36.695    0.002   54.803    0.004 {built-in method builtins.max}
        369218707   18.108    0.000   18.108    0.000 bpe.py:102(<lambda>)
                1    3.487    3.487  134.180  134.180 bpe.py:14(train_bpe)
          9927794    1.156    0.000    1.156    0.000 {method 'get' of 'dict' objects}

    ## owt_valid 학습 결과
    max pair 찾는 로직 개선 필요.
    owt는 TinyStories에 비해 단어가 다양하고 불규칙적인 데이터여서 merge가 병목이다.

    1. train with owt, none-heapq
        vocab_size: 32000
        special_tokens:['<|endoftext|>']
        num_chunks:16
        init vocab: 0.022ms
        pre-tokenization: 4957.090ms
        merge: 1911055.916ms
        elapsed time: 1916013.028ms

    2. train with owt, using heapq
        vocab_size: 32000
        special_tokens:['<|endoftext|>']
        num_chunks:16
        init vocab: 0.021ms
        pre-tokenization: 4702.632ms
        merge: 260860.510ms
        elapsed time: 265563.163ms
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--data", type=str, default="tiny")
    args = parser.parse_args()

    if args.data == "tiny":
        bpe.train_bpe_tinystories(vocab_size=10000, special_tokens=["<|endoftext|>"], num_chunks=args.chunks)
        with open("TinyStories_merges.pkl", "rb") as f:
            merges = pickle.load(f)
    elif args.data == "owt":
        bpe.train_bpe_expts_owt(vocab_size=32000, special_tokens=["<|endoftext|>"], num_chunks=args.chunks)
        with open("owt_merges.pkl", "rb") as f:
            merges = pickle.load(f)
    else:
        bpe.train_bpe_examples(vocab_size=10000, special_tokens=["<|endoftext|>"], num_chunks=args.chunks)
        with open("examples_merges.pkl", "rb") as f:
            merges = pickle.load(f)

    for s in merges[:21]:
        print(b"".join(s), s)
