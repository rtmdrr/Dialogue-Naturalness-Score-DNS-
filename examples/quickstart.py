"""A self-contained tour of the metric, using stub backends.

Run it with ``python examples/quickstart.py``. It needs nothing beyond the core
install: the entity extractor and emotion classifier are replaced with small
deterministic stubs so the example runs offline in a second.

For real use, swap the stubs for the bundled backends::

    from deep_persona_dns import SpacyEntityExtractor, load_nrc_english
    evaluator = DNSEvaluator(
        load_nrc_english(), entity_extractor=SpacyEntityExtractor()
    )
"""

from deep_persona_dns import (
    CallableEmotionVectorizer,
    CallableEntityExtractor,
    DNSEvaluator,
    Turn,
    dns_score,
    fit_baseline,
    naturalness_pvalue,
)

TOPICS = ("garden", "cello", "border collie", "night shift")
FEELINGS = {"happy", "sad", "love", "afraid", "delighted", "tired"}


def stub_extractor():
    return CallableEntityExtractor(
        lambda text: [t for t in TOPICS if t in text.lower()]
    )


def stub_vectorizer():
    def encode(texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            negative = any(w in lowered for w in ("sad", "afraid", "tired", "sighs"))
            positive = any(w in lowered for w in ("happy", "love", "delighted", "smiles"))
            vectors.append([0.9, 0.1] if negative else [0.1, 0.9] if positive else [0.5, 0.5])
        return vectors

    return CallableEmotionVectorizer(encode)


def human_dialogues():
    """A small stand-in for a corpus of human conversation.

    A baseline is only useful if it varies the way real dialogue varies. These
    differ in whether the listener takes the topic up explicitly, in how much
    affective language they use, and in how long the replies run -- which are
    the things the components measure.

    The two are varied *independently*. A corpus in which, say, every attentive
    reply also happened to be an emotional one would make those components
    perfectly correlated, the covariance near-singular, and every dialogue
    scored against it would come out impossibly far from the centroid.
    """
    openers = ("tell me more about it", "go on", "mm", "how long has that been true?")
    neutral = ("and since then?", "who else knows?", "what changed?", "i see")
    affective = (
        "i am happy you brought it up",
        "that sounds like a sad thing to carry",
        "you must be tired of explaining it",
        "i would love to hear the rest",
    )
    dialogues = []
    for topic_index, topic in enumerate(TOPICS):
        for reply_index, opener in enumerate(openers):
            acknowledges = reply_index % 2 == 0
            is_affective = topic_index % 2 == 0
            second = (
                f"the {topic} is part of it, then"
                if acknowledges
                else "that is a while to sit with something"
            )
            third = affective[reply_index] if is_affective else neutral[reply_index]
            warm_close = (topic_index // 2) % 2 == 0
            fourth = (
                "i would love to hear how it settles, however tired you are of it"
                if warm_close
                else "these things rarely have a clean beginning"
            )
            dialogues.append([
                Turn(f"i have been thinking about the {topic}", opener),
                Turn("it has been on my mind all week", second),
                Turn("it started a while ago", third),
                Turn("hard to say exactly", fourth),
            ])
    return dialogues


def main() -> None:
    evaluator = DNSEvaluator(FEELINGS, entity_extractor=stub_extractor())

    profiles = evaluator.profiles(human_dialogues())
    baseline = fit_baseline(profiles, evaluator.components)
    print(f"baseline: {baseline.n} dialogues over {list(baseline.components)}")
    spread = profiles.std(axis=0, ddof=1)
    print("  spread: " + "  ".join(
        f"{name}={value:.3f}" for name, value in zip(baseline.components, spread)
    ))
    print("  a baseline with little spread makes every dialogue look unnatural\n")

    attentive = [
        Turn("i have been thinking about the garden", "what has the garden been doing?"),
        Turn("the roses came back", "roses coming back is a happy sort of surprise"),
        Turn("i was tired of them last year", "being tired of a thing rarely lasts"),
        Turn("apparently not", "gardens are patient with us"),
    ]
    echoing = [
        Turn("i have been thinking about the garden", "i have been thinking about the garden"),
        Turn("the roses came back", "the roses came back"),
        Turn("i was tired of them last year", "i was tired of them last year"),
        Turn("apparently not", "apparently not"),
    ]

    print("component scores")
    for name, dialogue in (("attentive", attentive), ("echoing", echoing)):
        scores = evaluator.evaluate(dialogue).scores()
        rendered = "  ".join(
            f"{k}={v:.3f}" for k, v in scores.items() if v is not None
        )
        print(f"  {name:10s} {rendered}")

    candidates = evaluator.profiles([attentive, echoing])
    scores = dns_score(candidates, baseline)
    pvalues = naturalness_pvalue(candidates, baseline)
    print("\nDNS (1 = at the human centroid)")
    for name, score, pvalue in zip(("attentive", "echoing"), scores, pvalues):
        verdict = "not distinguishable" if pvalue > 0.05 else "distinguishable"
        print(f"  {name:10s} DNS={score:.4f}  p={pvalue:.4f}  {verdict}")

    print("\nadding affective congruence")
    full = DNSEvaluator(
        FEELINGS,
        entity_extractor=stub_extractor(),
        emotion_vectorizer=stub_vectorizer(),
    )
    for description, agent_turn in (
        ("speech and action agree", "i am so tired of it [sighs]"),
        ("speech and action differ", "i am delighted, truly [sighs]"),
    ):
        profile = full.evaluate([Turn("how has it been?", agent_turn)])
        print(f"  {description:24s} congruence={profile.scores()['congruence']:.3f}")


if __name__ == "__main__":
    main()
