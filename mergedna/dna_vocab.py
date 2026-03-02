from dataclasses import dataclass

@dataclass(frozen=True)
class DNAVocab:
    # Canonical bases + ambiguity + mask
    A: int = 0
    C: int = 1
    G: int = 2
    T: int = 3
    N: int = 4
    MASK: int = 5

    @property
    def size(self) -> int:
        return 6

VOCAB = DNAVocab()

def encode_dna_string(seq: str) -> list[int]:
    # Uppercase and map any non-ACGT characters to N
    out = []
    for ch in seq.upper():
        if ch == "A":
            out.append(VOCAB.A)
        elif ch == "C":
            out.append(VOCAB.C)
        elif ch == "G":
            out.append(VOCAB.G)
        elif ch == "T":
            out.append(VOCAB.T)
        else:
            out.append(VOCAB.N)
    return out