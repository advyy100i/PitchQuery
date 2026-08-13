"""Phase 3: build the two retrieval indexes over `possessions.token_string`.

Sparse (does the real work): TF-IDF over word n-grams (1,3). The vocabulary is
controlled — 14 actions x 15 zones x 8 modifier combinations — so n-grams of
tokens behave like phrases, and a sparse matrix over 67k possessions fits in RAM
and answers in milliseconds. Pickled to INDEX_DIR, loaded once at API startup.

Dense (catches paraphrase): all-MiniLM-L6-v2 into pgvector.

Run:
  python ingest/05_embed.py --sparse-only     # no torch needed, seconds
  python ingest/05_embed.py                   # both, then build the HNSW index
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from core import db  # noqa: E402
from core.config import EMBED_DIM, EMBED_MODEL, INDEX_DIR  # noqa: E402

SPARSE_PATH = INDEX_DIR / "tfidf.pkl"


def load_corpus(conn):
    """uids and token strings in a fixed order — row i of the matrix is uids[i]."""
    with conn.cursor() as cur:
        cur.execute("SELECT possession_uid, token_string FROM possessions "
                    "ORDER BY possession_uid")
        rows = cur.fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def build_sparse(uids, docs):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # tokenizer=str.split is not optional. The default token_pattern is
    # r"(?u)\b\w\w+\b", which would shred 'CROSS@F-R>' into 'cross' and 'f'
    # and throw away every modifier — the exact signal the grammar encodes.
    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=str.split,
        token_pattern=None,   # explicit: silences sklearn's "unused parameter" warning
        preprocessor=None,
        lowercase=False,
        ngram_range=(1, 3),
        min_df=2,             # an n-gram seen once cannot help retrieval
        sublinear_tf=True,    # a 60-token possession should not dominate on length
        norm="l2",
    )
    t0 = time.time()
    matrix = vec.fit_transform(docs)
    print(f"  tfidf: {matrix.shape[0]:,} x {matrix.shape[1]:,} "
          f"({matrix.nnz:,} nonzeros, {matrix.data.nbytes / 1e6:.0f} MB) "
          f"in {time.time() - t0:.1f}s")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPARSE_PATH, "wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": matrix, "uids": uids}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {SPARSE_PATH} ({SPARSE_PATH.stat().st_size / 1e6:.0f} MB)")
    return matrix


def build_dense(conn, uids, docs, batch: int = 512):
    from sentence_transformers import SentenceTransformer

    print(f"  loading {EMBED_MODEL} (CPU)")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    t0 = time.time()
    vectors = model.encode(docs, batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    print(f"  encoded {len(docs):,} strings in {time.time() - t0:.0f}s "
          f"-> {vectors.shape}")
    assert vectors.shape[1] == EMBED_DIM, f"expected {EMBED_DIM} dims"

    t0 = time.time()
    with conn.cursor() as cur:
        for i in range(0, len(uids), batch):
            chunk = [(f"[{','.join(f'{v:.6f}' for v in vec)}]", uid)
                     for uid, vec in zip(uids[i:i + batch], vectors[i:i + batch])]
            cur.executemany(
                "UPDATE possessions SET embedding = %s::vector WHERE possession_uid = %s",
                chunk)
            conn.commit()
            if (i // batch) % 20 == 0:
                print(f"    {min(i + batch, len(uids)):,}/{len(uids):,} written")
    print(f"  wrote embeddings in {time.time() - t0:.0f}s")


def build_hnsw(conn):
    """Built after the bulk insert on purpose — filling an existing HNSW index
    row by row is far slower than indexing a populated table."""
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS possessions_emb_idx ON possessions "
                    "USING hnsw (embedding vector_cosine_ops)")
    conn.commit()
    print(f"  hnsw index built in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse-only", action="store_true",
                    help="skip the MiniLM pass (no torch required)")
    ap.add_argument("--dense-only", action="store_true")
    args = ap.parse_args()

    conn = db.connect()
    uids, docs = load_corpus(conn)
    if not uids:
        print("no possessions — run ingest/04_build_possessions.py first")
        return
    print(f"corpus: {len(uids):,} possessions, "
          f"mean {np.mean([len(d.split()) for d in docs]):.1f} tokens")

    if not args.dense_only:
        print("sparse:")
        build_sparse(uids, docs)
    if not args.sparse_only:
        print("dense:")
        build_dense(conn, uids, docs)
        build_hnsw(conn)
    conn.close()


if __name__ == "__main__":
    main()
