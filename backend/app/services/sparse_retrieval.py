"""Sparse (keyword) retrieval with BM25 over the chunks stored in ChromaDB.

ChromaDB stays the single source of truth: the BM25 index is an in-memory copy
built from the stored chunk texts and metadata. It is rebuilt lazily, on the
first search after the store changed, detected two ways:
- `vector_store.index_version()`, bumped by every upload, delete and reset in this process;
- the stored chunk count, which also catches changes made by another process.

BM25 is implemented here rather than with `rank_bm25.BM25Okapi`, whose IDF is
zero for a term found in exactly half of the chunks (so a real match scores 0 on
small documents). This uses the Lucene/Elasticsearch IDF, which is always positive.
"""

import heapq
import logging
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.services import vector_store
from app.services.vector_store import StoredChunk

logger = logging.getLogger(__name__)

# Standard BM25 parameters: k1 limits how much repeating a word helps, b how much long chunks are penalised.
BM25_K1 = 1.5
BM25_B = 0.75

# Words too common to say anything about relevance. Kept short on purpose: words
# like "not", "no" or "after" can change the meaning of a policy clause.
STOPWORDS = frozenset(
    "a an and are as at be by can could do does did for from has have how i if in is it its "
    "may me might my of on or our shall should so than that the their them then there these "
    "they this those to was we were what when where which who whom why will with would you your".split()
)

# Words, numbers and identifiers. "7.3", "hr-27", "err_1042" and "v2/api" stay whole.
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[._/-][^\W_]+)*")
# Dots are not split: "7.3" must not match "3.7" or "3 days" through its parts.
TOKEN_SEPARATORS = re.compile(r"[_/-]")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens. Hyphenated, underscored and slashed identifiers are kept whole
    and also split into parts, so "HR-27" matches both "HR-27" and "HR 27"."""
    tokens: list[str] = []
    for word in TOKEN_PATTERN.findall(text.lower()):
        parts = TOKEN_SEPARATORS.split(word)
        if len(parts) > 1:
            tokens.append(word)
        tokens.extend(part for part in parts if _is_meaningful(part))
    return tokens


def _is_meaningful(token: str) -> bool:
    # Single letters (the "s" of "child's") carry nothing; single digits can ("3 days").
    return token not in STOPWORDS and (len(token) > 1 or token.isdigit())


@dataclass(frozen=True)
class SparseMatch:
    chunk: StoredChunk
    score: float  # raw BM25 score: only comparable with other scores for the same query


class BM25Index:
    def __init__(self, chunks: list[StoredChunk]) -> None:
        self.chunks = chunks
        self.lengths: list[int] = []
        # term -> [(chunk position, term frequency)]: only chunks containing a term are ever scored.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((position, frequency))
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def __len__(self) -> int:
        return len(self.chunks)

    def idf(self, term: str) -> float:
        containing = len(self.postings.get(term, ()))
        return math.log(1 + (len(self.chunks) - containing + 0.5) / (containing + 0.5))

    def search(self, query: str, top_k: int, document_id: str | None = None) -> list[SparseMatch]:
        """Chunks sharing at least one meaningful word with the query, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for position, frequency in postings:
                if document_id and self.chunks[position].metadata.get("document_id") != document_id:
                    continue
                length_norm = 1 - BM25_B + BM25_B * self.lengths[position] / (self.average_length or 1)
                scores[position] += idf * frequency * (BM25_K1 + 1) / (frequency + BM25_K1 * length_norm)

        # Ties keep storage order so results are deterministic.
        best = heapq.nsmallest(top_k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [SparseMatch(chunk=self.chunks[position], score=score) for position, score in best]


_cache: tuple[tuple[int, int], BM25Index] | None = None
_cache_lock = threading.Lock()


def get_sparse_index() -> BM25Index:
    """The BM25 index for the current store contents, rebuilt only when the store changed."""
    global _cache
    # Read the version before the chunks: a write that lands during the rebuild leaves
    # the cached key stale, so the next search rebuilds again instead of missing it.
    key = (vector_store.index_version(), vector_store.count_chunks())
    cached = _cache
    if cached is not None and cached[0] == key:
        return cached[1]

    with _cache_lock:
        if _cache is not None and _cache[0] == key:
            return _cache[1]
        index = BM25Index(vector_store.get_all_chunks())
        _cache = (key, index)
        logger.info("Built BM25 index: %d chunks, %d terms", len(index), len(index.postings))
        return index


def reset_sparse_index() -> None:
    global _cache
    with _cache_lock:
        _cache = None


def sparse_search(query: str, top_k: int, document_id: str | None = None) -> list[SparseMatch]:
    if top_k < 1 or not tokenize(query):
        return []  # an empty query, or one made only of stopwords, matches nothing
    index = get_sparse_index()
    if not len(index):
        return []
    return index.search(query, top_k, document_id)
