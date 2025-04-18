import pickle
import json
from collections import defaultdict, Counter
import math
from typing import Generic, TypeVar, Union, SupportsFloat, overload
import numpy as np
from tqdm import tqdm

class RunningAverage:
    def __init__(self):
        self.sum = 0
        self.n = 0
        
    def __add__(self, other):
        self.sum += other
        self.n += 1
        return self
        
    @property
    def avg(self):
        return self.sum / self.n

        
T = TypeVar('T', int, float)
        
class GramDict(dict[tuple, T], Generic[T]):
            
    def __init__(self, existing: Union[dict[tuple, T], None] = None, default: T = 0):
        self.default = default
        if existing is None:
            super(__class__, self).__init__()
        else:
            super(__class__, self).__init__(existing)
            
    
    def __missing__(self, __key: tuple) -> T:
        return self.default
        
    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError("Can only add another GramDict")
        return GramDict({gram: self[gram] + other[gram] for gram in (*self.keys(), *other.keys())})
    
    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError("Can only add another GramDict")
        
        for gram, val in other.items():
            self[gram] += val
        return self

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError("Can only multiply another GramDict")
        
        return GramDict({gram: self[gram] * other[gram] for gram in self.keys()})

    
    def __matmul__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError("Can only matmul another GramDict")
        
        elementwise_product = self * other
        return sum(elementwise_product.values())
    
    def magnitude(self):
        return math.sqrt(sum([val ** 2 for val in self.values()]))
    
    def apply(self, func):
        for gram, val in self.items():
            self[gram] = func(gram, val)


def build_ngrams(input_sequence: list[int], n: int) -> GramDict[int]:
    # all_grams = [NGram(input_sequence[k:k+n]) for k in range(len(input_sequence))]
    # grams = GramDict({gram: all_grams.count(gram) for gram in all_grams})
        
    # grams: defaultdict[NGram, int] = GramDict()
    # for k in range(len(input_sequence) - n + 1):
    #     grams[NGram(input_sequence[k:k+n])] += 1
    
    grams = GramDict(Counter([tuple(input_sequence[k:k+n]) for k in range(len(input_sequence)-n+1)]))
            
    return grams
    

def calculate_tf(gram_counts: GramDict[int]) -> GramDict[float]:    
    num_grams = sum(gram_counts.values())
    return GramDict({gram: count / 1.0 for gram, count in gram_counts.items()})

def el_min(a: GramDict[float], b: GramDict[float]) -> GramDict[float]:
    return GramDict({gram: min(a[gram], b[gram]) for gram in a.keys()})
    
class CIDERn:
    
    def __init__(self, n: int, d: bool) -> None:
        self.n = n
        self.d = d
        self.sigma = 6
        self.reset()
        
    def reset(self):
        self.ref_gram_tf: list[list[GramDict[float]]] = []
        self.cand_gram_tf: list[GramDict[float]] = []
        self.ref_len: list[list[int]] = []
        self.cand_len: list[int] = []
        self.gram_df: GramDict[int] = None
        
    def add_sample(self, candidate: list[int], references: list[list[int]]):
        # Calculate TF of g_k(c_i) for every k
        self.cand_gram_tf.append(calculate_tf(build_ngrams(candidate, self.n)))
        self.cand_len.append(len(candidate))
        
        # Caluclate TF of g_k(s_ij) for every k and j
        ref_gram_counts = [build_ngrams(ref, self.n) for ref in references]
        self.ref_gram_tf.append([calculate_tf(count) for count in ref_gram_counts])
        self.ref_len.append([len(ref) for ref in references])
            
    def _calc_gram_df(self):
        gram_df = GramDict()
        # for each ngram count the number of images where it appears in refs
        for refs in self.ref_gram_tf:
            for ngram in {ngram for ref in refs for ngram in ref.keys()}:
                gram_df[ngram] += 1
                
        self.gram_df = gram_df
        self.log_I = np.log(len(self.ref_gram_tf))
        
            
    def calc_one_sample_by_index(self, i: int):
        return self.calc_one_sample(self.cand_gram_tf[i], self.cand_len[i], self.ref_gram_tf[i], self.ref_len[i])
    
    def gram_dict_to_vec(self, gram_dict: GramDict[float]):
        vec = GramDict[float]()
        
        norm = 0.0
        for ngram, tf in gram_dict.items():
            df = np.log(max(1, self.gram_df[ngram])) # give ngram df of 1 if it does not appear
            idf = self.log_I - df
            vec[ngram] = tf * idf
            norm += vec[ngram] ** 2
            
        return vec, np.sqrt(norm)
    
    def cosine_similarity(self, a: GramDict[float], b: GramDict[float], a_len: int, b_len: int, a_mag: float, b_mag: float):
        sim = 0
        
        if self.d:
            for ngram, tfidf in a.items():
                sim += min(tfidf, b[ngram]) * b[ngram]
        else:
            for ngram, tfidf in a.items():
                sim += tfidf * b[ngram]
            
        # if l2 norm is 0, dot product is also 0
        if a_mag == 0 or b_mag == 0:
            return 0
        sim /= (a_mag * b_mag)
        if self.d:
            sim *= 10 * np.exp(-((a_len - b_len) ** 2) / (2 * self.sigma ** 2))
        return sim
            
    def calc_one_sample(self, cand_gram_tf: GramDict[float], cand_len: int, ref_gram_tf: list[GramDict[float]], ref_len: list[int]):                           
        g_c, c_mag = self.gram_dict_to_vec(cand_gram_tf)
        
        scores = []
        for i in range(len(ref_gram_tf)):
            g_si, si_mag = self.gram_dict_to_vec(ref_gram_tf[i])
            
            score = self.cosine_similarity(g_c, g_si, cand_len, ref_len[i], c_mag, si_mag)
            scores.append(score)
    
        avg_score = sum(scores) / len(scores)
        return avg_score
    
    def calc_sample_direct(self, candidate: list[int], references: list[list[int]]):
        # Calculate TF of g_k(c_i) for every k
        cand_gram_tf = calculate_tf(build_ngrams(candidate, self.n))
        cand_len = len(candidate)
        
        # Caluclate TF of g_k(s_ij) for every k and j
        ref_gram_counts = [build_ngrams(ref, self.n) for ref in references]
        ref_gram_tf = [calculate_tf(count) for count in ref_gram_counts]
        ref_len = [len(ref) for ref in references]
        
        # Calc sample, without updating idf
        if self.gram_df is None:
            raise Exception("Need to set idf first!")
        
        return self.calc_one_sample(cand_gram_tf, cand_len, ref_gram_tf, ref_len)
        
    
    def calc_all_samples(self):
        if self.gram_df is None:
            self._calc_gram_df()
        return np.array([self.calc_one_sample(self.cand_gram_tf[i], self.cand_len[i], self.ref_gram_tf[i], self.ref_len[i]) for i in range(len(self.cand_gram_tf))])
    
    
    def __len__(self):
        return len(self.cand_gram_tf)
    

class CIDER:
    
    def __init__(self, n: int = 4, d: bool = True) -> None:
        self.n = n
        self.cider_ns = [CIDERn(i, d) for i in range(1, n+1)]
        self.d = d

    def reset(self):
        for n in range(len(self.cider_ns)):
            self.cider_ns[n].reset()
                    
    def add_sample(self, candidate: list[int], references: list[list[int]]):
        for cider_n in self.cider_ns:
            cider_n.add_sample(candidate, references)
            
    def calc_one_sample_by_index(self, i: int):
        return sum([cider_n.calc_one_sample_by_index(i) for cider_n in self.cider_ns]) / len(self.cider_ns)
    
    def calc_all_samples(self, silent=False):
        iterator = range(len(self.cider_ns[0]))
        if not silent:
            iterator = tqdm(iterator)
            
        for cider_n in self.cider_ns:
            if cider_n.gram_df is None:
                cider_n._calc_gram_df()
        return np.array([self.calc_one_sample_by_index(i) for i in iterator])
    
    def calc_score(self, keepCache = False, silent=False):
        scores = self.calc_all_samples(silent)
        return scores.mean(), scores.std()
    
    def calc_sample_direct(self, candidate: list[int], references: list[list[int]]):
        return sum([cider_n.calc_sample_direct(candidate, references)  for cider_n in self.cider_ns]) / len(self.cider_ns)
    
    def export_df(self, path):
        data = {
            'dfs': [cider_n.gram_df for cider_n in self.cider_ns],
            'log_I': self.cider_ns[0].log_I
        }
        with open(path, "wb") as file:
            pickle.dump(data, file)
            
    def load_df(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)
        for idx, df in enumerate(data['dfs']):
            self.cider_ns[idx].gram_df = df
            self.cider_ns[idx].log_I = data['log_I']