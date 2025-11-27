from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path

class BPETrainer:
    def __init__(self, vocab_size=32000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq

    def train(self, data_files, out_path="tokenizer.json"):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_freq
        )
        tokenizer.train(data_files, trainer)

        tokenizer.save(out_path)
        print(f"Saved tokenizer → {out_path}")
        return out_path
