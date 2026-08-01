import regex as re
import pstats
import cProfile
from pstats import SortKey
from collections import defaultdict
from multiprocessing import Pool
from functools import wraps


def profile(enabled=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.strip_dirs()
            stats.sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(30)
            return result

        return wrapper
    return decorator

class BPETrainer:
    def __init__(self):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def train(self, input_path, target_vocab_size, special_tokens) \
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(input_path, "r", encoding="utf-8") as F:
            text = F.read()
            token_counter = 0
            tokenid2token = {}
            token2tokenid = {}
            tokenid2freq = defaultdict(int)
            pretokenizers = self.__pretokenize(text, special_tokens)

            for pretokenizer in pretokenizers:
                for token in pretokenizer:
                    token = tuple(bytes([b]) for b in token)
                    if not token in token2tokenid:
                        tokenid2token[token_counter] = token
                        token2tokenid[token] = token_counter
                        tokenid2freq[token_counter] = 1
                        token_counter += 1
                    else:
                        tokenid = token2tokenid[token]
                        tokenid2freq[tokenid] += 1

            vocabs, merges = self.__merge(target_vocab_size, tokenid2token, tokenid2freq, special_tokens)

            return vocabs, merges

    @profile(enabled=False)
    def __pretokenize(self, text, special_tokens=None):
        if special_tokens is None:
            return [re.finditer(self.pattern, text)]

        special_tokens = [re.escape(token) for token in special_tokens]
        documents = re.split("|".join(special_tokens), text)
        with Pool(8) as p:
            documents = p.map(self.pretoken_single_doc, documents)
        return documents

    @profile(enabled=False)
    def __merge(self, target_vocab_size, tokenid2token, tokenid2freq, special_tokens):
        vocabs = {}
        merges = []
        pair2tokenid = defaultdict(set)
        pair2freq = defaultdict(int)

        for i in range(256):
            vocabs[i] = bytes([i])
        for i in range(256, len(special_tokens) + 256):
            vocabs[i] = special_tokens[i - 256].encode("utf-8")

        for tokenid, token in tokenid2token.items():
            for pair in zip(token, token[1:]):
                pair2tokenid[pair].add(tokenid)
                pair2freq[pair] += tokenid2freq[tokenid]

        while len(vocabs) < target_vocab_size:
            max_freq = max(pair2freq.values())
            max_pairs = {pair: freq for pair, freq in pair2freq.items() if freq == max_freq}
            current_merge = sorted(max_pairs, reverse=True)[0]
            merges.append(current_merge)
            vocabs[len(vocabs)] = current_merge[0] + current_merge[1]
            tokenids = pair2tokenid[current_merge].copy()
            for tokenid in tokenids:
                old_token = tokenid2token[tokenid]
                new_token = self.__merge_token(old_token, current_merge)
                tokenid2token[tokenid] = new_token
                for old_pair in zip(old_token, old_token[1:]):
                    pair2tokenid[old_pair].discard(tokenid)
                    pair2freq[old_pair] -= tokenid2freq[tokenid]
                for new_pair in zip(new_token, new_token[1:]):
                    pair2tokenid[new_pair].add(tokenid)
                    pair2freq[new_pair] += tokenid2freq[tokenid]

        return vocabs, merges

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
        return tuple(merged_token)

    def pretoken_single_doc(self, document):
        tokens = [m.group() for m in re.finditer(self.pattern, document)]
        tokens = list(map(lambda token: token.encode("utf-8"), tokens))
        return tokens


if __name__ == "__main__":
    bpe_trainer = BPETrainer()
    vocabs, merges = bpe_trainer.train("test.txt", 258, ["<|endoftext|>"])

    print(vocabs)
    print(merges)