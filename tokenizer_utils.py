import os
import pickle
import re
from collections import Counter

class DynamicBPETokenizer:
    def __init__(self, vocab_size_max=10000):
        self.vocab_size_max = vocab_size_max
        self.vocab = {}
        self.encoder = {}
        self.decoder = {}
        
    def train_from_text(self, text):
        print("Đang tạo Tokenizer từ dữ liệu...")
        words = re.findall(r'\S+|\s+', text)
        word_counts = Counter(words)
        chars = sorted(list(set(text)))
        self.decoder = {i: ch for i, ch in enumerate(chars)}
        self.encoder = {ch: i for i, ch in enumerate(chars)}       
        current_vocab_size = len(chars)
        final_vocab_size = min(self.vocab_size_max, current_vocab_size + 2000) 
        splits = {word: [char for char in word] for word in word_counts}    
        
        def get_stats():
            pairs = Counter()
            for word, freq in word_counts.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i+1]] += freq
            return pairs
        
        def merge_pair(pair_to_merge):
            p1, p2 = pair_to_merge
            for word in word_counts:
                symbols = splits[word]
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == p1 and symbols[i+1] == p2:
                        symbols[i:i+2] = [p1 + p2]
                    else:
                        i += 1
        
        while current_vocab_size < final_vocab_size:
            pairs = get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            merge_pair(best_pair)            
            new_token = best_pair[0] + best_pair[1]
            self.encoder[new_token] = current_vocab_size
            self.decoder[current_vocab_size] = new_token
            current_vocab_size += 1           
        print(final_vocab_size)
        print(f"-> Hoàn thành tạo Tokenizer. Kích thước Vocab thực tế: {len(self.encoder)}")
    
    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            match = None
            for length in range(max(len(k) for k in self.encoder.keys()), 0, -1):
                substr = text[i:i+length]
                if substr in self.encoder:
                    match = substr
                    break
            if match:
                tokens.append(self.encoder[match])
                i += len(match)
            else:
                tokens.append(self.encoder.get(text[i], 0))
                i += 1
        return tokens

    def decode(self, tokens):
        return "".join([self.decoder.get(t, "") for t in tokens])

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'encoder': self.encoder, 'decoder': self.decoder}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.encoder = data['encoder']
            self.decoder = data['decoder']