from typing import Iterable, Iterator
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re




class BPE_Encoder:
    vocab = None
    vocab_rev = None
    merges = None
    special_tokens = None
    num_processes = 10
    
    def __apply_merge(self, item:bytes):
        tokens = [bytes([b]) for b in item]
        new_tokens = [];
        for merge in self.merges:
            i = 0
            while i < len(tokens)-1:
                if tokens[i] == merge[0] and tokens[i+1] == merge[1]:
                    if i!=len(tokens)-2:
                        new_tokens = tokens[:i] + [tokens[i]+tokens[i+1]] + tokens[i+2:]
                    else:
                        new_tokens = tokens[:i] + [tokens[i]+tokens[i+1]]
                    tokens = new_tokens
                i+=1
        res = list();
        for token in tokens:
            res.append(self.vocab_rev[token])
        return res
            
            
        
    
    def __init__(self, vocab: dict[int, bytes] , merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_rev = {v:k for k,v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        return
    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        return
    
    def encode(self, text: str) -> list[int]:
        res = []
        cache = dict();
        if self.special_tokens == None:
            text_binary = text.encode('utf-8')
            # map 会把任务分发给 8 个进程并行执行，并等待所有进程返回结果
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for match in re.finditer(PAT, text):
                item = match.group(0).encode("utf-8")
                if not item in cache:
                    # TODO apply_merge
                    merge_res = self.__apply_merge(item)
                    cache[item] = merge_res
                res = res + cache[item]
            return res
        else:
            sorted_specials = sorted(self.special_tokens, key=lambda s: len(s), reverse=True)
            escaped_specials = [re.escape(token) for token in sorted_specials]
            # escaped_specials = [re.escape(token) for token in self.special_tokens]
            
            pattern_str = f"({'|'.join(escaped_specials)})"
            pattern = re.compile(pattern_str)
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            
            results = []
            last_idx = 0
            
            # 3. 使用 finditer 遍历所有匹配到的 special tokens
            for match in pattern.finditer(text):
                start, end = match.span()
                
                # 如果当前特殊 token 前面有普通的文本，先把它提取出来
                if start > last_idx:
                    for match1 in re.finditer(PAT, text[last_idx:start]):
                        item = match1.group(0).encode("utf-8")
                        if not item in cache:
                            merge_res = self.__apply_merge(item)
                            cache[item] = merge_res
                        res = res + cache[item]
                if match.group().encode("utf-8") in self.vocab_rev:
                    res = res + [self.vocab_rev[match.group().encode("utf-8")],]
                    # print(self.vocab_rev[match.group().encode("utf-8")])
                else:
                    print("have not such token:",match.group()," in vocab")
                last_idx = end
                
            # 4. 别忘了处理最后一个特殊 token 后面的剩余文本
            if last_idx < len(text):
                for match1 in re.finditer(PAT, text[last_idx:]):
                    item = match1.group(0).encode("utf-8")
                    if not item in cache:
                        merge_res = self.__apply_merge(item)
                        cache[item] = merge_res
                    res = res + cache[item]
            return res
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            for idx in self.encode(line):
                yield idx
    
    def decode(self, ids: list[int]) -> str:
        res = b"".join(self.vocab[i] for i in ids)
        return bytes(res).decode("utf-8",  errors="replace")