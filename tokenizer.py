import re
from collections import Counter
from nltk.tokenize import word_tokenize


class SimpleTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<start>", "<end>"]

    def __init__(self):
        # Dictionary mappings between words and indices
        self.word2idx = {}
        self.idx2word = {}

    def _tokenize(self, text):
        return word_tokenize(text.lower())

    def build_vocab(self, texts):
        # Count all tokens in the texts
        word_counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counter.update(tokens)

        # Create vocabulary with special tokens at the beginning
        vocabulary = self.SPECIAL_TOKENS + sorted(word_counter.keys())

        # Create mapping dictionaries
        self.word2idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text):
        # Add start and end tokens
        tokens = ["<start>"] + self._tokenize(text) + ["<end>"]

        # Get the unknown token index (default to 1 if not found)
        unk_idx = self.word2idx.get("<unk>", 1)

        # Convert tokens to indices
        return [self.word2idx.get(word, unk_idx) for word in tokens]

    def decode(self, indices):
        # Convert indices to tokens
        tokens = [self.idx2word.get(idx, "<unk>") for idx in indices]

        # Join tokens into a sentence
        sentence = " ".join(tokens)

        # Fix punctuation spacing
        sentence = re.sub(r"\s+([?.!,\"'])", r"\1", sentence)

        # Remove special tokens and clean up whitespace
        sentence = sentence.replace("<pad>", "").replace("<unk>", "").strip()

        return sentence

    def save_vocab(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for word, idx in self.word2idx.items():
                f.write(f"{word}\t{idx}\n")

    def load_vocab(self, file_path):
        self.word2idx = {}

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word, idx = line.strip().split("\t")
                self.word2idx[word] = int(idx)

        # Rebuild the reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}