import regex as re
import pstats
import cProfile
from pstats import SortKey
from collections import defaultdict
from multiprocessing import Pool

class Tokenizer:
    def __init__(self):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


    def train(self, input_path, target_vocab_size, special_tokens) \
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        F = open(input_path, "r", encoding="utf-8")
        text = F.read()
        merges = []
        
        token2freq = defaultdict(int)
        vocabs = {}

        for i in range(256):
            vocabs[i] = bytes([i])
        for i in range(256, len(special_tokens) + 256):
            vocabs[i] = special_tokens[i - 256].encode("utf-8")
        
        pretokenizers = self.__pretokenize(text, special_tokens)
        for pretokenizer in pretokenizers:
            for token in pretokenizer:
                token2freq[tuple(bytes([b]) for b in token)] += 1

        while len(vocabs) < target_vocab_size:
            pairs = defaultdict(int)
            new_token2freq = defaultdict(int)

            for byte_chr, freq in token2freq.items():
                for pair in zip(byte_chr, byte_chr[1:]):
                    pairs[pair] += freq

            max_freq = max(pairs.values())
            max_pairs = {pair: freq for pair, freq in pairs.items() if freq == max_freq}
            current_merge = sorted(max_pairs, reverse=True)[0]
            merges.append(current_merge)
            vocabs[len(vocabs)] = current_merge[0] + current_merge[1]

            for token, freq in token2freq.items():
                merged_token = self.__merge_token(token, current_merge)
                new_token2freq[tuple(merged_token)] += freq

            token2freq = new_token2freq

        return vocabs, merges

    def __pretokenize(self, text, special_tokens=None):
        if special_tokens is None:
            return [re.finditer(self.pattern, text)]

        special_tokens = [re.escape(token) for token in special_tokens]
        documents = re.split("|".join(special_tokens), text)
        with Pool(8) as p:
            documents = p.map(self.pretoken_single_doc, documents)
        return documents

    def __merge_token(self, token, merge):
        merged_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == merge:
                merged_token.append(token[i] + token[i + 1])
                i += 2
            else:
                merged_token.append(token[i])
                i += 1
        return merged_token

    def pretoken_single_doc(self, document):
        tokens = [m.group() for m in re.finditer(self.pattern, document)]
        tokens = list(map(lambda token: token.encode("utf-8"), tokens))
        return tokens


if __name__ == "__main__":
    profiler = cProfile.Profile()
    tokenizer = Tokenizer()

    profiler.enable()
    vocabs, merges = tokenizer.train("test.txt", 258, ["<|endoftext|>"])
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)

    print(vocabs)
    print(merges)