"""Deterministic corpus iteration and prompt-overlap exclusion."""
import collections
import pyarrow.parquet as pq
from common import STORAGE,normalize

def rows(split):
    directory = STORAGE / 'datasets/smoltalk/data/all'
    files = sorted(directory.glob(f'{split}-*.parquet'))
    assert len(files) == (9 if split == 'train' else 1), (split, files)
    index = 0
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                yield index, row
                index += 1

class PromptFilter:
    def __init__(self, prompts):
        self.texts = {normalize(p) for p in prompts}
        self.shingles = []
        self.index = collections.defaultdict(set)
        for text in sorted(self.texts):
            words = text.split()
            grams = set(zip(*(words[i:] for i in range(5))))
            j = len(self.shingles)
            self.shingles.append(grams)
            for gram in grams: self.index[gram].add(j)

    def overlaps(self, text):
        text = normalize(text)
        if text in self.texts: return True
        words = text.split()
        grams = set(zip(*(words[i:] for i in range(5))))
        counts = collections.Counter(j for gram in grams for j in self.index.get(gram, ()))
        for j, intersection in counts.items():
            other = self.shingles[j]
            union = len(grams) + len(other) - intersection
            if union and intersection / union >= .8: return True
            if min(len(grams), len(other)) >= 10 and intersection / min(len(grams), len(other)) >= .9:
                return True
        return False
