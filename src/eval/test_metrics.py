"""Smoke tests. Run: python -m src.eval.test_metrics"""

from .metrics import evaluate
from .schema import Reference, Transcript, Word


def perfect_match() -> None:
    words = [Word(text=t, start=i * 0.5, end=i * 0.5 + 0.4) for i, t in enumerate(["hello", "world"])]
    ref = Reference(track_id="t1", language="en", words=words)
    hyp = Transcript(language="en", words=words)
    r = evaluate(ref, hyp)
    assert r.text.wer == 0.0
    assert r.timing.mean_abs_offset == 0.0
    assert r.timing.coverage == 1.0
    assert r.halluc.spurious_word_rate == 0.0
    print("perfect_match OK", r.as_row())


def single_substitution() -> None:
    ref = Reference(
        track_id="t2",
        language="en",
        words=[
            Word(text="hello", start=0.0, end=0.4),
            Word(text="world", start=0.5, end=0.9),
        ],
    )
    hyp = Transcript(
        language="en",
        words=[
            Word(text="hello", start=0.0, end=0.4),
            Word(text="word", start=0.5, end=0.9),
        ],
    )
    r = evaluate(ref, hyp)
    assert 0.0 < r.text.wer <= 1.0
    assert r.timing.coverage == 0.5  # only "hello" matched
    print("single_substitution OK", r.as_row())


def hallucination_in_silence() -> None:
    ref = Reference(
        track_id="t3",
        language="en",
        words=[Word(text="hi", start=0.0, end=0.3)],
        silence_intervals=[(1.0, 5.0)],
    )
    hyp = Transcript(
        language="en",
        words=[
            Word(text="hi", start=0.0, end=0.3),
            Word(text="ghost", start=2.0, end=2.5),
            Word(text="phantom", start=3.0, end=3.5),
        ],
    )
    r = evaluate(ref, hyp)
    assert r.halluc.spurious_words == 2
    assert abs(r.halluc.spurious_word_rate - 2 / 3) < 1e-9
    print("hallucination_in_silence OK", r.as_row())


if __name__ == "__main__":
    perfect_match()
    single_substitution()
    hallucination_in_silence()
    print("\nall tests passed")
