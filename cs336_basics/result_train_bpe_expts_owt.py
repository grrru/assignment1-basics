import json

with open("artifacts/owt_vocab.json") as f:
    """
    Problem (train_bpe_expts_owt):  BPE Training on OpenWebText (2 points)

    (a) What is the longest token in the vocabulary? Does it make sense?

    25822, ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ
    25836, ----------------------------------------------------------------
    31274, ————————————————
    10900, --------------------------------
    15947, ________________________________
    16885, ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ
    25146, ================================
    28585, ................................
    31162, ********************************
    15279, ————————
    23327,  disproportionately
    24268,  telecommunications
    28274,  environmentalists
    14284,  responsibilities
    16284,  unconstitutional
    25698,  cryptocurrencies
    26073,  disproportionate
    27038,  misunderstanding
    28492,  counterterrorism
    30211,  characterization

    (b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.

    Open Web Text의 가장 긴 토큰은 깨진 인코딩이다. 
    TinyStories에는 Open Web Text의 잡다한 토큰이 나타나지 않는다. Web 특성인듯.
    어휘 성격도 경제,기술 <-> 동화 중심 차이가 있다.
    7160,  accomplishment
    9143,  disappointment
    9379,  responsibility
    3228,  uncomfortable
    3515,  compassionate
    5319,  understanding
    6386,  neighbourhood
    6497,  Unfortunately
    6874,  determination
    7756,  encouragement
    8626,  unfortunately
    8699,  congratulated
    8868,  extraordinary
    9095,  granddaughter
    256, <|endoftext|>
    3338,  disappointed
    3769,  enthusiastic
    4360,  accidentally
    4376,  refrigerator
    4472,  veterinarian
    """
    vocab: dict[str, int] = json.load(f)
    li = [((item[1], item[0])) for item in vocab.items()]

    li.sort(key=lambda x: len(x[1]), reverse=True)

    for el in li[:20]:
        print(f"{el[0]}, {el[1]}")
