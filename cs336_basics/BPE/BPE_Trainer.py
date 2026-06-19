import os
from pathlib import Path
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from collections import Counter
import time


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(args: tuple[str, int, int,list[str]]) -> dict:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""";
    file_path, start, end, special_tokens = args
    res = dict();
    with open(file_path,"rb") as f:
        f.seek(start);
        chunk_bytes = f.read(end - start);
        chunk_text = chunk_bytes.decode(encoding="utf-8", errors="ignore");
        patt = r'(?:';
        patt = patt + r'|'.join(re.escape(d) for d in special_tokens);
        patt = patt + r')';
        chunk_texts = re.split(patt, chunk_text)
        for ct in chunk_texts:
            for match in re.finditer(PAT, ct):
                item = match.group(0).encode("utf-8")
                if item in res:
                    res[item] += 1
                else:
                    res[item] = 1
    return res

# def process_chunk(args: tuple[str, int, int,list[str]]) -> dict:
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""";
#     # 在最前面加上 (?!<\|endoftext\|>)，意思是：匹配以下任何规则前，先确保它不是 <|endoftext|>
#     # PAT = re.compile(r"""(?!\s*<\|endoftext\|>)(?:'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)""")
#     file_path, start, end, special_tokens = args
#     res = dict();
#     with open(file_path,"rb") as f:
#         f.seek(start);
#         chunk_bytes = f.read(end - start);
#         chunk_text = chunk_bytes.decode(encoding="utf-8", errors="ignore");
#         patt = r'(?:';
#         patt = patt + r'|'.join(re.escape(d) for d in special_tokens);
#         patt = patt + r')';
#         chunk_text = re.split(patt, chunk_text)
#         for ct in chunk_text:
#             for match in re.finditer(PAT, ct):
#                 item = match.group(0).encode("utf-8")
#                 if item in res:
#                     res[item] += 1
#                 else:
#                     res[item] = 1
#     return res

# def process_chunk(args: tuple[str, int, int,list[str]]) -> dict:
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""";
#     file_path, start, end, special_tokens = args
#     res = dict();
#     with open(file_path,"rb") as f:
#         f.seek(start);
#         chunk_bytes = f.read(end - start);
#         chunk_text = chunk_bytes.decode(encoding="utf-8", errors="ignore");
#         patt = "|".join(re.escape(token) for token in special_tokens)
#         chunk_text = re.split(patt, chunk_text)
#         chunk_text = ''.join(chunk_text)
#         for match in re.finditer(PAT, chunk_text):
#             item = match.group(0).encode("utf-8")
#             if item in res:
#                 res[item] += 1
#             else:
#                 res[item] = 1
#     return res


def Pre_tokenization(input_path:str, special_tokens:list[str]):
    ## Usage
    # Use a real data file so the example cell can run without manual edits.
    data_path = Path(input_path)
    if not data_path.exists():
        print(f"File not found: {input_path}")
        return;
    with data_path.open("rb") as f:
        num_chunk = 100
        num_processes = 10
        b_special_tokens = [item.encode('utf-8') for item in special_tokens]
        boundaries = find_chunk_boundaries(f, num_chunk, b_special_tokens[0])

        # 每个元素为: (文件路径, 起始字节, 结束字节)
        tasks = [
            (str(data_path), boundaries[i], boundaries[i+1], special_tokens) 
            for i in range(len(boundaries) - 1)
        ]
        print(f"开始并行处理，总共 {num_processes} 个进程...");
        
        global_counter = dict()
        i = 0
        while i < num_chunk:
            with Pool(processes=num_processes) as pool:
                # map 会把任务分发给 8 个进程并行执行，并等待所有进程返回结果
                results = pool.map(process_chunk, tasks)
                    
                for res_i in  results:
                    for item in res_i:
                        if item in global_counter:
                            global_counter[item] += res_i[item];
                        else:
                            global_counter[item] = res_i[item];
                i+=num_processes
        return global_counter

def Vocabulary_init(special_tokens: list[str]):
    res = { i:st.encode("utf-8") for i,st in enumerate(special_tokens)}
    for i in range(256):
        res[len(res)] = bytes([i])
    return res;

from cs336_basics.BPE import merges_algo
def Merges(tokens:dict,vocab_size: int,len_special_tokens:int):
    trained_vocab, merges_history = merges_algo.run_algorithm(tokens,vocab_size-len_special_tokens);
    return trained_vocab, merges_history
import time
def BPE_Tokenizer(input_path: str, vocab_size: int, special_tokens: list[str])-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocabulary (list[str]) and convert to dict[int, bytes]
    t1 = time.time();
    vocab = Vocabulary_init(special_tokens)

    # pre-tokenize the input file
    token_counts = Pre_tokenization(input_path,special_tokens)
    t2 = time.time();
    print("time of Pre_tokenization:",t2-t1,"ms")

    # get merges (may be None for now)
    print(len(vocab))
    trained_vocab, merges_history = Merges(token_counts, vocab_size, len(vocab));
    for item in trained_vocab:
        vocab[len(vocab)] = item
    t2 = time.time();
    print("using time:",t2-t1,"ms");

    return vocab, merges_history


BPE_Tokenizer("/home/jarch/vscode_resp/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",2000,["<|endoftext|>"])
