# Dialogue Naturalness Score (DNS)

A reference-free metric for how far an agent's dialogue departs from human
conversational behaviour.

DNS computes a small behavioural profile from an agent's turns, compares it with
the distribution of the same profile over a corpus of human dialogue, and
reports both a score and a statistical test. It needs no reference response and
no human rating at inference time — only a corpus of human dialogue to calibrate
against.

This is the reference implementation for the metric described in *Deep Persona:
A Psychologically Grounded Architecture and Evaluation Framework for Role-Playing
Agents and Simulations* ([arXiv:2609.22255](https://arxiv.org/abs/2609.22255)).

---

## Install

```bash
pip install deep-persona-dns
```

The core install computes the lexicon-based components and the whole scoring
layer. The backends that need models are optional extras:

```bash
pip install "deep-persona-dns[english]"     # spaCy entities, NRC lexicon
pip install "deep-persona-dns[hebrew]"      # Hebrew lexicon, Hebrew NER
pip install "deep-persona-dns[congruence]"  # emotion classifier
```

For English you will also need a spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

## Quickstart

```python
from deep_persona_dns import (
    DNSEvaluator, SpacyEntityExtractor, Turn,
    dns_score, fit_baseline, load_nrc_english, naturalness_pvalue,
)

evaluator = DNSEvaluator(
    load_nrc_english(),
    entity_extractor=SpacyEntityExtractor(),
)

# A dialogue is a list of Turns, each pairing a user utterance with the
# agent response to it.
dialogue = [
    Turn(user="i have been thinking about the garden",
         agent="what has the garden been doing?"),
    Turn(user="the roses came back",
         agent="roses coming back is a happy sort of surprise"),
]

# Calibrate on human dialogue, then score.
baseline = fit_baseline(
    evaluator.profiles(human_dialogues), evaluator.components
)
scores = dns_score(evaluator.profiles([dialogue]), baseline)
pvalues = naturalness_pvalue(evaluator.profiles([dialogue]), baseline)
```

`examples/quickstart.py` runs the whole flow offline with stub backends.

### From the command line

```bash
dns fit-baseline human.jsonl -o baseline.json
dns score agent.jsonl --baseline baseline.json -o scores.jsonl
```

Both read JSON Lines, one dialogue per line:

```json
{"dialogue_id": "d1", "turns": [{"user": "...", "agent": "..."}]}
```

Nonverbal actions go in square brackets inside the agent text:
`"i'm fine [avoids eye contact]"`.

---

## What it measures

Four components are computed from the agent's turns. The first three are
bounded in `[0, 1]`; the fourth is a rate and is unbounded above.

| component | high value means |
|---|---|
| **Pragmatic alignment** | the agent neither mirrors the user nor repeats itself |
| **Joint attention** | the agent takes up entities and topics the user introduces |
| **Affective congruence** | what the agent says matches what it is described as doing |
| **Emotional expression** | the agent draws on a range of affective vocabulary |

Affective congruence requires nonverbal actions in the transcript. Without
them the profile has three components, which is reported by
`evaluator.components` rather than assumed.

The profile is compared with the human baseline by squared Mahalanobis
distance, which gives:

- **the score**, `DNS = exp(-λ·d²)`, equal to 1 at the human centroid and
  falling towards 0 as the profile departs from it;
- **the test**, since `d²` follows a χ² distribution with as many degrees of
  freedom as there are components. A non-significant result means the dialogue
  is not distinguishable from the baseline on these components.

λ defaults to `ln2 / χ²₀.₉₅,df`, which puts DNS at 0.5 on the significance
boundary whatever the profile width. For three components that is 0.089.

---

## Choosing a baseline

Everything DNS reports is relative to the baseline, which makes the choice of
baseline corpus as consequential as any parameter. Two things matter:

**It must resemble what you will score.** A baseline of short task-oriented
exchanges will call long discursive conversation unnatural, and correctly so by
its own standard. Match register, length and domain.

**It must vary.** The metric measures distance from a distribution, so the
distribution needs spread. A corpus built from one template produces a baseline
whose components barely move, and then every dialogue scored against it comes
out impossibly far from the centroid. `fit_baseline` refuses a corpus with no
variation at all, but it cannot detect a corpus that merely has too little, so
inspect the per-component spread before trusting a baseline.

---

## Configuration

Defaults reproduce the configuration reported in the paper. Where the metric
admits more than one reasonable reading, both are available:

```python
from deep_persona_dns import DNSEvaluator, EvaluatorConfig

config = EvaluatorConfig(
    acknowledgement_window="next",   # or "current_or_next"
    emotion_counting="per_turn",     # or "distinct"
    intensity_scope="adjacent",      # or "turn"
    overlap_measure="rouge_l",       # or "jaccard"
    self_repetition_window=5,
)
```

| option | default | alternative |
|---|---|---|
| `acknowledgement_window` | the agent's next turn | also accept the same turn |
| `emotion_counting` | terms per turn, repeats count | distinct terms per dialogue |
| `intensity_scope` | modifiers near an emotion term | any modifier in the turn |
| `overlap_measure` | ROUGE-L | Jaccard |

`fit_baseline` takes `estimator="shrinkage"` (default), `"scale_invariant"` or
`"empirical"`. Shrinkage is well defined where the sample covariance is not.
The scale-invariant estimator regularises the correlation structure while
leaving each component's own variance alone, which is worth using when the
components differ in variance by orders of magnitude.

---

## Extending it

The two model-backed steps are protocols, so any annotator can be substituted.

```python
from deep_persona_dns import CallableEntityExtractor

def annotate(utterance: str) -> list[str]:
    """Entities introduced in this utterance, from any source you like."""
    ...

evaluator = DNSEvaluator(
    lexicon, entity_extractor=CallableEntityExtractor(annotate)
)
```

The paper proposes a language model as the primary annotator for joint
attention, with named entity recognition as an alternative. `CallableEntityExtractor`
is the seam for the former; results are cached per utterance. The bundled
backends implement the latter, and are the default because they are
deterministic, offline and free — properties a metric intended to be
reproducible should have.

The emotion vectorizer works the same way through `CallableEmotionVectorizer`.

Lexicons are ordinary sets of strings. `load_from_file` reads one word per
line, so any word list can be used in place of the bundled loaders.

---

## Languages

English and Hebrew are configured. Other languages need a `LanguageConfig`, a
lexicon, and an entity backend:

```python
from deep_persona_dns import LanguageConfig, get_language

config = LanguageConfig(code="xx", intensifiers=frozenset({...}))
```

The bundled lexicon loaders read from separately distributed packages rather
than copying word lists into this repository, so each resource keeps its own
licence and citation. English uses the NRC Emotion Lexicon (Mohammad and
Turney, 2013). Hebrew uses the emotion categories of the Hebrew Psychological
Lexicons (Shapira et al., CLPsych 2021), distributed under CC BY-SA 4.0.

---

## Citing

If you use this metric, please cite the paper:

```bibtex
@article{dror2026deeppersona,
  title   = {Deep Persona: A Psychologically Grounded Architecture and
             Evaluation Framework for Role-Playing Agents and Simulations},
  author  = {Dror, Rotem and Elyoseph, Zohar and Haber, Yuval and
             Refoua, Elad and Ayalon, Oshrat and Solomon, Adir},
  journal = {arXiv preprint arXiv:2609.22255},
  year    = {2026}
}
```

Please also cite whichever lexicon you use.

---

## Licence

MIT. See `LICENSE`.
