# Configuration reference

Every option, what it changes, and what it defaults to.

**The defaults reproduce the configuration reported in the paper.** Where the
metric admits more than one reasonable reading, both readings are implemented
and the default follows the paper. Nothing here needs to be set to use the
metric as published.

Configuration lives in three places:

| | set by | affects |
|---|---|---|
| component behaviour | `EvaluatorConfig`, passed to `DNSEvaluator` | the profile |
| covariance estimation | `estimator=` on `fit_baseline` | the reference distribution |
| score decay | `lam=` / `quantile=` on `dns_score` | the score, not the ranking |

---

## 1. Component options

```python
from deep_persona_dns import DNSEvaluator, EvaluatorConfig

evaluator = DNSEvaluator(
    lexicon,
    entity_extractor=extractor,
    config=EvaluatorConfig(acknowledgement_window="current_or_next"),
)
```

### Pragmatic alignment

| option | default | meaning |
|---|---|---|
| `alpha` | `1/3` | weight of user–agent overlap in the penalty |
| `beta` | `1/3` | weight of self-repetition |
| `gamma` | `1/3` | weight of the echolalia rate |
| `echolalia_threshold` | `0.65` | overlap above which a turn counts as echolalia |
| `self_repetition_window` | `5` | how many previous agent responses to compare against |
| `overlap_measure` | `"rouge_l"` | `"rouge_l"` or `"jaccard"` |
| `clamp_pragmatics` | `True` | floor the score at 0 |

The three weights must sum to 1, and the component is
`1 − (α·overlap + β·self_repetition + γ·echolalia_rate)`.

`self_repetition_window` bounds how far back lexical fixation is detected. A
larger window catches an agent that recycles a phrase every few turns; a
smaller one only catches immediate repetition.

`clamp_pragmatics=False` lets the score go negative, which happens when all
three penalties are high at once. Useful if you want to distinguish degrees of
badness below zero; the default keeps the component on the same `[0, 1]` scale
as the others.

### Joint attention

| option | default | meaning |
|---|---|---|
| `acknowledgement_window` | `"next"` | where an entity may be acknowledged |

- `"next"` — the entity must appear in the agent response *after* the turn that
  introduced it. This is the definition in the paper.
- `"current_or_next"` — the immediate response also counts, on the view that
  answering at once is no worse than answering a turn later.

`"current_or_next"` is the more permissive of the two and cannot score a
dialogue lower than `"next"`.

### Emotional expression

| option | default | meaning |
|---|---|---|
| `emotion_weight` | `0.7` | weight of emotion-term diversity |
| `intensity_weight` | `0.3` | weight of intensity-modifier diversity |
| `emotion_counting` | `"per_turn"` | how terms are counted |
| `intensity_scope` | `"adjacent"` | how modifiers are attributed to a turn |
| `intensity_window` | `3` | tokens either side that count as adjacent |

The two weights must sum to 1.

`emotion_counting` decides what "diversity" means:

- `"per_turn"` — each turn contributes the number of matching tokens it
  contains, averaged over turns. A word used in five turns counts five times.
  This is the formula in the paper, and reads as affective *density*.
- `"distinct"` — how many different terms the dialogue uses at all, divided by
  its length. A word used in five turns counts once, so it reads as affective
  *range*.

`intensity_scope` decides whether position matters:

- `"adjacent"` — a modifier counts only within `intensity_window` tokens of an
  emotion term, so "very happy" counts and a stray "very" does not.
- `"turn"` — every modifier in the turn counts regardless of position.

Neither diversity is bounded above: both are counts normalised by dialogue
length, so this component is on a different scale from the other three. That is
intentional — the Mahalanobis step standardises the profile against the
baseline — but it matters when reading raw component values.

### Affective congruence

| option | default | meaning |
|---|---|---|
| `congruence_similarity` | `"cosine"` | `"cosine"` or `"pearson"` |
| `skip_uninformative_pairs` | `True` | set aside pairs the emotion model had no opinion on |

A pair is uninformative when the emotion model returns a near-uniform
probability vector for the speech or the action: the classifier found nothing,
so the angle between the two vectors reflects noise rather than agreement.
Skipped pairs are counted and reported on the result as `n_uninformative`, so
it is always visible how much of the dialogue was actually scored.

This component is only computed when the evaluator has an emotion vectorizer.
Without one the profile is three components wide. Check `evaluator.components`
rather than assuming — it returns
`("pragmatics", "joint_attention", "emotion")` or the four-component tuple with
`"congruence"` in third place.

---

## 2. Covariance estimation

```python
baseline = fit_baseline(profiles, components, estimator="shrinkage")
```

| estimator | behaviour |
|---|---|
| `"shrinkage"` *(default)* | Ledoit–Wolf shrinkage of the covariance |
| `"scale_invariant"` | shrinkage applied to the correlation matrix, per-component variance restored |
| `"empirical"` | the sample covariance, pseudo-inverted if singular |

**`"shrinkage"`** is well defined where the sample covariance is not, which
happens readily with a modest baseline or a component that barely varies.

**`"scale_invariant"`** matters when the components differ in variance by
orders of magnitude. Ledoit–Wolf shrinks toward a scaled identity whose scale
is the mean eigenvalue, so it is set by the largest-variance components; a
component whose baseline variance is much smaller has its variance pulled up
toward that target, which reduces how much it can contribute to `d²`. The
scale-invariant estimator standardises first, shrinks the correlation matrix —
which is scale-free — and then restores each component's own standard
deviation. The correlation structure is still regularised; the relative scale
of the components is not touched.

Since the components here are on genuinely different scales (three are bounded
in `[0, 1]`, emotional expression is an unbounded rate), this is worth checking
on your own baseline. Compare the two fits and see whether the ranking changes.

**`"empirical"`** is the literal reading of the definition and is singular
whenever a component is constant across the baseline, in which case the
pseudo-inverse is used.

### What a baseline must look like

`fit_baseline` refuses a corpus whose profiles show no variation at all, since
the reference distribution would be a point and every dialogue would score a
perfect 1. It cannot detect a corpus that merely has *too little* variation,
which produces the opposite failure: every dialogue lands impossibly far from
the centroid. Inspect the spread before trusting a fit:

```python
profiles = evaluator.profiles(human_dialogues)
print(profiles.std(axis=0, ddof=1))
```

Two other properties are worth checking. The baseline should resemble what you
will score in register, length and domain. And the components should vary
*independently* — if the corpus happens to couple two of them, the covariance
becomes near-singular and distances inflate.

Rows with a missing component are dropped from the fit rather than imputed; the
number actually used is recorded as `baseline.n`.

---

## 3. Degrees of freedom, λ and the test

```python
scores  = dns_score(profiles, baseline)                  # λ from the profile width
scores  = dns_score(profiles, baseline, lam=0.089)       # or pinned explicitly
pvalues = naturalness_pvalue(profiles, baseline)
passes  = is_indistinguishable(profiles, baseline, alpha=0.05)
```

`DNS = exp(−λ·d²)`. By default λ is `ln2 / χ²₀.₉₅,df`, derived from the number
of components in the baseline, which puts DNS at exactly 0.5 on the 0.05
significance boundary whatever the profile width:

| components | λ |
|---|---|
| 3 | 0.0887 |
| 4 | 0.0731 |

`lambda_for_df(df, quantile)` computes it directly. Pass `quantile=` to move
the half-way point, or `lam=` to pin a particular constant — which is what you
want when reproducing results computed with a specific value.

**λ does not change any ranking.** `exp(−λ·d²)` is monotone in `d²`, so
correlations and orderings are identical for any positive λ. It sets where on
the `(0, 1]` range the scores sit, nothing more.

`naturalness_pvalue` and `is_indistinguishable` take `df=` to override the
degrees of freedom, and the latter takes `alpha=` for the significance level.

Two cautions on the test. It asks whether a profile is *typical* of the
baseline, not whether the dialogue is good. And failing to reject the null is
not evidence for it: a small baseline or noisy components make rejection
unlikely whatever the dialogue, so read the pass rate alongside `baseline.n`.

For a sufficiently distant dialogue `exp(−λ·d²)` underflows to zero. If you
need to compare two dialogues that both score 0, compare `mahalanobis_sq`
instead.

---

## 4. Backends

### Entity extraction

| class | constructor |
|---|---|
| `SpacyEntityExtractor` | `(model="en_core_web_sm", min_chars=3)` |
| `TransformerEntityExtractor` | `(model, device=None, min_chars=3)` |
| `CallableEntityExtractor` | `(annotator, min_chars=3, cache=True)` |

`min_chars` discards candidates shorter than three characters, which are rarely
referential. `device` is a pipeline device index; `None` selects CUDA when
available.

`CallableEntityExtractor` wraps any function from an utterance to a list of
entity strings. The paper proposes a language model as the primary annotator
for joint attention, with named entity recognition as the alternative; this is
the seam for the former. Results are cached per utterance, so a repeated
utterance costs one call. Pass `cache=False` to disable.

`build_entity_extractor(language)` constructs the default for a language.

### Emotion vectorizers

| class | constructor |
|---|---|
| `TransformerEmotionVectorizer` | `(model, batch_size=16, device=None, max_length=512)` |
| `EnsembleEmotionVectorizer` | `(model_template, emotions=PLUTCHIK_EMOTIONS, batch_size=16, device=None, max_length=512, positive_index=1)` |
| `CallableEmotionVectorizer` | `(encoder)` |

`TransformerEmotionVectorizer` reads the model's own `problem_type` and applies
softmax or the logistic function accordingly.

`EnsembleEmotionVectorizer` stacks one binary classifier per emotion, taking
each model's positive-class probability. `model_template` is a format string
containing `{emotion}`. `positive_index` says which output column is the
positive class.

`build_emotion_vectorizer(language)` constructs the default for a language.

### Language defaults

| | English | Hebrew |
|---|---|---|
| entity model | `en_core_web_sm` | `avichr/heBERT_NER` |
| emotion model | `bhadresh-savani/distilbert-base-uncased-emotion` | `avichr/hebEMO_{emotion}` |
| intensifiers | 15 | 13 |

Build a `LanguageConfig` for any other language; `code`, `intensifiers` and
optionally the two default model names are all it needs.

### Lexicons

| function | source |
|---|---|
| `load_nrc_english()` | NRC Emotion Lexicon, via `nrclex` |
| `load_hebrew_psychological()` | emotion categories of the Hebrew Psychological Lexicons, via `hepsylex` |
| `load_from_file(path)` | UTF-8 text, one word per line |

Neither bundled lexicon is copied into this repository; both are read from the
package that distributes them, so each keeps its own licence and citation. A
lexicon is just a set of lowercase strings, so any word list works.
`load_from_file` ignores blank lines and lines beginning with `#`, so a file
can carry a provenance note at the top.

---

## 5. Command line

```bash
dns fit-baseline human.jsonl -o baseline.json
dns score agent.jsonl --baseline baseline.json -o scores.jsonl
```

Shared by both commands:

| flag | default | meaning |
|---|---|---|
| `--language` | `en` | language configuration |
| `--lexicon` | bundled loader | path to a lexicon file, one word per line |
| `--congruence` | off | compute affective congruence, giving a four-component profile |
| `--acknowledgement-window` | `next` | `next` or `current_or_next` |
| `--emotion-counting` | `per_turn` | `per_turn` or `distinct` |
| `--intensity-scope` | `adjacent` | `adjacent` or `turn` |
| `--overlap-measure` | `rouge_l` | `rouge_l` or `jaccard` |
| `--self-repetition-window` | `5` | previous responses compared against |

`fit-baseline` only:

| flag | default | meaning |
|---|---|---|
| `-o`, `--output` | *required* | where to write the baseline |
| `--estimator` | `shrinkage` | `shrinkage`, `scale_invariant` or `empirical` |

`score` only:

| flag | default | meaning |
|---|---|---|
| `--baseline` | *required* | baseline to score against |
| `-o`, `--output` | stdout | where to write the scores |
| `--lam` | derived | decay constant |
| `--verbose` | off | include per-component diagnostics |

Scoring reads the component list from the baseline and refuses to run when the
configuration cannot produce it, rather than silently scoring a different
profile. The component options must therefore match those used to fit the
baseline.

### Formats

Input, one dialogue per line:

```json
{"dialogue_id": "d1", "turns": [{"user": "...", "agent": "..."}]}
```

`dialogue_id` is optional and defaults to the line number. Nonverbal actions go
in square brackets inside the agent text.

Output, one dialogue per line:

```json
{"dialogue_id": "d1", "dns": 0.87, "mahalanobis_sq": 1.4, "p_value": 0.71,
 "components": {"pragmatics": 0.91, "joint_attention": 0.5, "emotion": 0.35}}
```

A dialogue missing a component gets `null` for the score and that component,
rather than being dropped silently.

---

## 6. Reproducing the published configuration

Every default is already the published setting, so:

```python
evaluator = DNSEvaluator(lexicon, entity_extractor=extractor)
baseline = fit_baseline(profiles, evaluator.components)
scores = dns_score(candidate_profiles, baseline)
```

Explicitly, that is α = β = γ = 1/3, `echolalia_threshold` 0.65, emotion
weights 0.7 and 0.3, λ = 0.089 for a three-component profile, Ledoit–Wolf
shrinkage, ROUGE-L overlap, acknowledgement in the next turn, per-turn emotion
counting and adjacent intensity attribution.

The one setting the paper does not fix is the self-repetition window, which is
5 here.
