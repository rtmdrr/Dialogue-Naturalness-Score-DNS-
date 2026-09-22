"""Command-line interface.

Two commands, matching the two stages of using the metric::

    dns fit-baseline human.jsonl -o baseline.json
    dns score agent.jsonl --baseline baseline.json -o scores.jsonl

Both read JSON Lines, one dialogue per line::

    {"dialogue_id": "d1", "turns": [{"user": "...", "agent": "..."}, ...]}

``dialogue_id`` is optional and defaults to the line number. Nonverbal actions
go in square brackets inside the agent text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .baseline import ESTIMATORS, HumanBaseline, fit_baseline
from .emotion import build_emotion_vectorizer
from .entities import build_entity_extractor
from .evaluator import DNSEvaluator, EvaluatorConfig
from .languages import get_language
from .lexicons import load_from_file, load_hebrew_psychological, load_nrc_english
from .score import dns_score, mahalanobis_sq, naturalness_pvalue
from .turns import Turn


def read_dialogues(path: str) -> Tuple[List[str], List[List[Turn]]]:
    """Read dialogues from a JSON Lines file."""
    ids: List[str] = []
    dialogues: List[List[Turn]] = []
    if not Path(path).is_file():
        raise SystemExit(f"no such file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}: line {number} is not valid JSON: {exc}")
            turns = record.get("turns")
            if not isinstance(turns, list) or not turns:
                raise SystemExit(f"{path}: line {number} has no 'turns' list")
            try:
                parsed = [Turn(user=t["user"], agent=t["agent"]) for t in turns]
            except (TypeError, KeyError) as exc:
                raise SystemExit(
                    f"{path}: line {number} has a turn without 'user' and 'agent': {exc}"
                )
            ids.append(str(record.get("dialogue_id", number)))
            dialogues.append(parsed)
    if not dialogues:
        raise SystemExit(f"{path}: no dialogues found")
    return ids, dialogues


def load_lexicon(language_code: str, path: Optional[str]) -> Set[str]:
    """Load the emotion lexicon named on the command line."""
    if path:
        return load_from_file(path)
    if language_code == "he":
        return load_hebrew_psychological()
    return load_nrc_english()


def build_evaluator(args) -> DNSEvaluator:
    language = get_language(args.language)
    lexicon = load_lexicon(language.code, args.lexicon)
    extractor = build_entity_extractor(language)
    vectorizer = build_emotion_vectorizer(language) if args.congruence else None
    config = EvaluatorConfig(
        acknowledgement_window=args.acknowledgement_window,
        emotion_counting=args.emotion_counting,
        intensity_scope=args.intensity_scope,
        overlap_measure=args.overlap_measure,
        self_repetition_window=args.self_repetition_window,
    )
    return DNSEvaluator(
        lexicon,
        entity_extractor=extractor,
        language=language,
        emotion_vectorizer=vectorizer,
        config=config,
    )


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="JSON Lines file of dialogues")
    parser.add_argument("--language", default="en", help="language code (default: en)")
    parser.add_argument(
        "--lexicon",
        help="emotion lexicon file, one word per line; defaults to the bundled "
             "loader for the language",
    )
    parser.add_argument(
        "--congruence", action="store_true",
        help="compute affective congruence, giving a four-component profile "
             "(requires nonverbal actions in the transcripts)",
    )
    parser.add_argument(
        "--acknowledgement-window", default="next", choices=("next", "current_or_next"),
        help="where joint attention may be acknowledged (default: next)",
    )
    parser.add_argument(
        "--emotion-counting", default="per_turn", choices=("per_turn", "distinct"),
        help="how emotion terms are counted (default: per_turn)",
    )
    parser.add_argument(
        "--intensity-scope", default="adjacent", choices=("adjacent", "turn"),
        help="how intensity modifiers are attributed (default: adjacent)",
    )
    parser.add_argument(
        "--overlap-measure", default="rouge_l", choices=("rouge_l", "jaccard"),
        help="similarity used for user-agent overlap (default: rouge_l)",
    )
    parser.add_argument(
        "--self-repetition-window", type=int, default=5,
        help="how many previous agent responses to compare against (default: 5)",
    )


def command_fit_baseline(args) -> int:
    evaluator = build_evaluator(args)
    ids, dialogues = read_dialogues(args.input)
    print(f"scoring {len(dialogues)} baseline dialogues ...", file=sys.stderr)

    components = tuple(evaluator.components)
    profiles = evaluator.profiles(dialogues, components)
    usable = int((~np.isnan(profiles).any(axis=1)).sum())
    if usable < len(dialogues):
        print(
            f"note: {len(dialogues) - usable} dialogue(s) missing a component "
            "and excluded from the fit",
            file=sys.stderr,
        )

    baseline = fit_baseline(profiles, components, estimator=args.estimator)
    Path(args.output).write_text(
        json.dumps(baseline.to_dict(), indent=2), encoding="utf-8"
    )
    print(
        f"baseline fitted on {baseline.n} dialogues over {list(components)} "
        f"({args.estimator}) -> {args.output}",
        file=sys.stderr,
    )
    return 0


def read_baseline(path: str) -> HumanBaseline:
    """Read a fitted baseline, reporting a usable message if it cannot be."""
    if not Path(path).is_file():
        raise SystemExit(f"no such baseline file: {path}")
    try:
        return HumanBaseline.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(f"{path} is not a baseline written by 'dns fit-baseline': {exc}")


def command_score(args) -> int:
    baseline = read_baseline(args.baseline)
    args.congruence = "congruence" in baseline.components
    evaluator = build_evaluator(args)

    missing = [c for c in baseline.components if c not in evaluator.components]
    if missing:
        raise SystemExit(
            f"the baseline needs components {missing} that this configuration "
            "cannot produce; pass --congruence, or fit a baseline without them"
        )

    ids, dialogues = read_dialogues(args.input)
    print(f"scoring {len(dialogues)} dialogues ...", file=sys.stderr)

    results = evaluator.evaluate_many(dialogues)
    profiles = np.vstack([r.as_vector(baseline.components) for r in results])
    complete = ~np.isnan(profiles).any(axis=1)

    distances = np.full(len(profiles), np.nan)
    scores = np.full(len(profiles), np.nan)
    pvalues = np.full(len(profiles), np.nan)
    if complete.any():
        subset = profiles[complete]
        distances[complete] = mahalanobis_sq(subset, baseline)
        scores[complete] = dns_score(subset, baseline, lam=args.lam)
        pvalues[complete] = naturalness_pvalue(subset, baseline)

    lines = []
    for index, dialogue_id in enumerate(ids):
        record: Dict[str, object] = {
            "dialogue_id": dialogue_id,
            "dns": None if np.isnan(scores[index]) else float(scores[index]),
            "mahalanobis_sq": None if np.isnan(distances[index]) else float(distances[index]),
            "p_value": None if np.isnan(pvalues[index]) else float(pvalues[index]),
            "components": {
                name: (None if np.isnan(value) else float(value))
                for name, value in zip(baseline.components, profiles[index])
            },
        }
        if args.verbose:
            record["diagnostics"] = results[index].as_dict()
        lines.append(json.dumps(record, ensure_ascii=False))

    output = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"wrote {len(lines)} scores -> {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)

    if complete.any():
        passing = float((pvalues[complete] > 0.05).mean())
        print(
            f"mean DNS {np.nanmean(scores):.4f} over {int(complete.sum())} dialogues; "
            f"{passing:.1%} not distinguishable from the baseline at alpha=0.05",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dns",
        description="Dialogue Naturalness Score: score dialogues against a human baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit-baseline", help="estimate the human reference distribution"
    )
    _add_shared(fit)
    fit.add_argument("-o", "--output", required=True, help="where to write the baseline")
    fit.add_argument(
        "--estimator", default="shrinkage", choices=ESTIMATORS,
        help="covariance estimator (default: shrinkage)",
    )
    fit.set_defaults(func=command_fit_baseline)

    score = subparsers.add_parser("score", help="score dialogues against a baseline")
    _add_shared(score)
    score.add_argument("--baseline", required=True, help="baseline file to score against")
    score.add_argument("-o", "--output", help="where to write scores (default: stdout)")
    score.add_argument(
        "--lam", type=float, default=None,
        help="decay constant; derived from the number of components when omitted",
    )
    score.add_argument(
        "--verbose", action="store_true", help="include per-component diagnostics"
    )
    score.set_defaults(func=command_score)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
