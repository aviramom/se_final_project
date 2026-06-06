# fmeval/core/datasets/ — Dataset abstractions

`Dataset` is the ABC that wraps a benchmark in its original file format and exposes a
uniform interface. Adding a new benchmark means adding one subclass here — nothing else
in the system changes.

---

## Files

```
datasets/
  CLAUDE.md
  base.py          ← Dataset ABC
  squad.py         ← SQuAD (text QA benchmark)
  etth1.py         ← ETTh1 (time-series benchmark)
  ...              ← one file per benchmark
```

---

## Dataset ABC (`base.py`)

```python
from abc import ABC, abstractmethod
from typing import Iterator, Literal
from fmeval.core.sample import Sample

class Dataset(ABC):
    """
    Wraps a benchmark in its original format and yields standardized Samples.
    Never alters the raw benchmark files — standardization happens in code here.
    """

    @property
    @abstractmethod
    def modality(self) -> Literal["text", "time_series"]: ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in the registry and stored in results."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """Yield one Sample per benchmark record. Lazy loading preferred."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in the dataset (used for progress tracking)."""
        ...
```

---

## Concrete subclass pattern

Each benchmark file must:

1. Subclass `Dataset`.
2. Accept the path to the raw benchmark file in `__init__` — **do not copy or
   mutate the file**.
3. Parse the original format lazily in `__iter__` (avoid loading the whole file into
   memory).
4. Map the benchmark's native fields to `Sample.input` and `Sample.target` in the
   format the `ModelWrapper` expects (see `core/sample.py` for the contract).
5. Set `modality` to match the benchmark type.

---

## Data Immutability constraint

Standardization is done entirely in `__iter__`. If a concrete subclass writes,
copies, or transforms the raw file on disk, that is a bug. The raw benchmark files
in `data/` are read-only from the perspective of this codebase.
