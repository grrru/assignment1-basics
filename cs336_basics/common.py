def make_safety_byte_dicts() -> tuple[dict[int, str], dict[str, int]]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return (dict(zip(bs, characters)), dict(zip(characters, bs)))


(BYTE_TO_CHAR, CHAR_TO_BYTE) = make_safety_byte_dicts()
