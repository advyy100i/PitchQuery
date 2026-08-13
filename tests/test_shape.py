"""Shape matching — the pure logic behind drawing a possession on the pitch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.retrieval import is_subsequence, reduce_path  # noqa: E402


def test_reduce_collapses_consecutive_repeats():
    assert reduce_path("D-C D-C M-C M-C M-C F-C".split()) == ["D-C", "M-C", "F-C"]


def test_reduce_keeps_a_zone_revisited_later():
    # Going out to the wing and coming back is a different journey from staying.
    assert reduce_path("M-C M-R M-C".split()) == ["M-C", "M-R", "M-C"]


def test_subsequence_allows_gaps():
    path = "D-C M-C M-LI F-C".split()
    assert is_subsequence(["D-C", "M-C", "F-C"], path)


def test_subsequence_respects_order():
    path = "F-C M-C D-C".split()
    assert not is_subsequence(["D-C", "M-C", "F-C"], path)


def test_shape_longer_than_path_cannot_match():
    assert not is_subsequence(["D-C", "M-C", "F-C"], ["D-C"])


def test_coverage_prefers_the_whole_journey():
    """The scoring rule the ranking depends on.

    A possession whose entire reduced path is the drawn shape must score above
    one where the shape is a fragment — that is what stops a 40-touch move
    winning because three zones happen to line up inside it.
    """
    shape = ["D-C", "M-C", "F-C"]
    exact = reduce_path("D-C D-C M-C F-C F-C".split())
    sprawling = reduce_path("F-L M-L D-C M-C F-C M-R F-R M-C D-R".split())

    assert is_subsequence(shape, exact) and is_subsequence(shape, sprawling)
    assert len(shape) / len(exact) > len(shape) / len(sprawling)
