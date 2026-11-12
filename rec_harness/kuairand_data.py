"""Research-only leave-two-out cache; never derive it from the shorter benchmark cache."""
from pathlib import Path
import json


def make_research_cache(source, output, history=1024, provenance=None):
    from longseq_bench.data import make_cache
    # The reference cache keeps H+2 events. Ask for one extra, then declare H.
    # Training excludes two holdouts and still has H input/label pairs.
    meta = make_cache(source, output, history + 1, provenance)
    meta.update(max_history=history, retained_events=history + 3,
                protocol='kuairand-research-v1', split='per-user-leave-two-out',
                train_excludes=2, validation_offset=-2, final_offset=-1)
    from rec_harness.core import write_json
    write_json(Path(output)/'metadata.json', meta)
    return meta


class ResearchDataset:
    def __init__(self, cache, split):
        import numpy as np
        if split not in {'train', 'validation', 'final'}:
            raise ValueError('unknown research partition')
        self.root, self.split = Path(cache), split
        self.meta = json.loads((self.root/'metadata.json').read_text())
        self.history = self.meta['max_history']
        if self.meta.get('protocol') != 'kuairand-research-v1' or self.meta.get('retained_events', 0) < self.history + 3:
            raise ValueError('research cache must retain history+3 events')
        self.ids = np.load(self.root/'ids.npy', mmap_mode='r')
        self.times = np.load(self.root/'times.npy', mmap_mode='r')
        self.lengths = np.load(self.root/'lengths.npy', mmap_mode='r')
        if self.ids.shape[1] < self.history + 3 or (self.lengths < 4).any():
            raise ValueError('research requires four events per user and history+3 cache width')

    def __len__(self): return len(self.lengths)

    def __getitem__(self, index):
        import numpy as np
        import torch
        n, h = int(self.lengths[index]), self.history
        stop = n - 2 if self.split == 'train' else n - 1 if self.split == 'validation' else n
        start = max(0, stop - h - 1)
        sequence = self.ids[index, start:stop]
        timestamps = self.times[index, start:stop]
        history, length = sequence[:-1], len(sequence) - 1
        ids = torch.zeros(h, dtype=torch.long)
        times = torch.zeros(h + 1, dtype=torch.long)
        labels = torch.zeros(h, dtype=torch.long)
        ids[:length] = torch.from_numpy(history.copy())
        times[:length + 1] = torch.from_numpy(timestamps.copy())
        if self.split == 'train': labels[:length] = torch.from_numpy(sequence[1:].copy())
        return {'ids': ids, 'times': times, 'labels': labels, 'length': torch.tensor(length),
                'target': torch.tensor(int(sequence[-1])), 'user_index': torch.tensor(index)}
