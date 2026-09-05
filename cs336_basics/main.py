import pickle

from cs336_basics import bpe

if __name__ == "__main__":
    """
    1. non-chunking
        init vocab: 0.026ms
        pre-tokenization: 216829.838ms
        elapsed time: 247645.444ms
    2. chunking & parallelize pre-tokenization
        init vocab: 0.024ms
        pre-tokenization: 73145.086ms
        elapsed time: 104329.446ms

    (b) Profile your code. What part of the tokenizer training process takes the most time?
        pre_tokenization에서 가장 많은 시간을 사용한다.(poll은 worker가 기다린 시간)
        merge 단계에서 가장 빈번한 pair를 찾는 시간이 그 다음 대부분을 차지
 
        ncalls  tottime  percall  cumtime  percall filename:lineno(function)
               19   73.646    3.876   73.646    3.876 {method 'poll' of 'select.poll' objects}
            15326   36.695    0.002   54.803    0.004 {built-in method builtins.max}
        369218707   18.108    0.000   18.108    0.000 bpe.py:102(<lambda>)
                1    3.487    3.487  134.180  134.180 bpe.py:14(train_bpe)
          9927794    1.156    0.000    1.156    0.000 {method 'get' of 'dict' objects}
    """
    bpe.train_bpe_tinystories()

    with open("TinyStories_merges.pkl", "rb") as f:
        merges = pickle.load(f)

        for s in merges[:21]:
            print(b"".join(s), s)
