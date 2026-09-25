#!/usr/bin/env python3
"""Blind multi-model coding of the 50 cumulative CRAV vignette states.

The coding models see only one cumulative vignette state at a time plus the
frozen codebook. They never receive GPT-4o allocations, transition effects,
scenario/level identifiers, Figure 1, or the existing theory map.

Default three-provider coder panel:
  openai:gpt-5.6-sol
  anthropic:claude-opus-5-5
  typesafe:jev-latest

An OpenAI/Anthropic-only panel can instead be supplied with --coders.

API keys are loaded from a local .env file by default (without overriding
already-set environment variables): OPENAI_API_KEY, ANTHROPIC_API_KEY, and,
when used, TYPESAFE_API_KEY. The .env file should be excluded from version
control. Results are written incrementally so interrupted runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv

DEFAULT_CODERS = [
    "openai:gpt-5.6-sol",
    "anthropic:claude-opus-5-5",
    "typesafe:jev-latest",
]
DEFAULT_SEED = 20260925


class ContextRating(BaseModel):
    responsibility_controllability: int = Field(ge=1, le=7)
    need_vulnerability: int = Field(ge=1, le=7)
    external_constraint_coercion: int = Field(ge=1, le=7)
    mitigating_circumstances_motive: int = Field(ge=1, le=7)
    corrective_prosocial_effort: int = Field(ge=1, le=7)
    confidence: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1, max_length=500)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blindly code CRAV vignette states with multiple AI models.")
    p.add_argument("--input-dir", type=Path, default=Path("input_data"))
    p.add_argument("--pattern", default="crav_vignette_*_raw.csv")
    p.add_argument("--codebook", type=Path, default=Path(__file__).with_name("crav_context_coding_codebook.json"))
    p.add_argument("--env-file", type=Path, default=Path(".env"),
                   help="Local dotenv file containing provider API keys. Existing environment variables take precedence.")
    p.add_argument("--output", type=Path, default=Path("data/supplementary_data_s4_blinded_context_codings.csv"))
    p.add_argument("--manifest", type=Path, default=Path("checks/blinded_context_coding_manifest.csv"),
                   help="Local 50-state mapping used for audit/merge; never sent to coding APIs.")
    p.add_argument("--order-manifest", type=Path, default=Path("checks/blinded_context_coding_order.csv"),
                   help="Recorded random presentation order for each coder; never sent as metadata to coding APIs.")
    p.add_argument("--metadata", type=Path, default=Path("checks/blinded_context_coding_metadata.json"))
    p.add_argument("--coders", nargs="+", default=DEFAULT_CODERS,
                   help="Coder specs as provider:model. Supported providers: openai, anthropic, typesafe.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--prepare-only", action="store_true",
                   help="Create the blinded manifest and metadata without calling any API.")
    p.add_argument("--overwrite", action="store_true",
                   help="Discard an existing output file instead of resuming it.")
    return p.parse_args()


def sha256_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def stable_seed(base: int, text: str) -> int:
    h = hashlib.sha256(f"{base}|{text}".encode()).hexdigest()
    return int(h[:12], 16)


def parse_scenario_level(question_id: str) -> tuple[int, int]:
    m = re.search(r"Scenario_(\d+)(?:_V\d+)?_Level_(\d+)", str(question_id))
    if not m:
        raise ValueError(f"Cannot parse scenario/level from question id: {question_id!r}")
    return int(m.group(1)), int(m.group(2))


def load_states(input_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {input_dir.resolve()}")
    frames = [pd.read_csv(f, usecols=["question id", "question"]) for f in files]
    x = pd.concat(frames, ignore_index=True).drop_duplicates()
    parsed = x["question id"].map(parse_scenario_level)
    x[["scenario", "level"]] = pd.DataFrame(parsed.tolist(), index=x.index)
    x = x.drop_duplicates(["scenario", "level"]).sort_values(["scenario", "level"]).reset_index(drop=True)
    if len(x) != 50 or x["scenario"].nunique() != 10 or x["level"].nunique() != 5:
        raise ValueError(f"Expected exactly 50 unique scenario-level states; found {len(x)}")
    x["narrative"] = x["question"].astype(str).str.strip()
    x["item_hash"] = x["narrative"].map(lambda s: sha256_bytes(s.encode()))
    return x[["scenario", "level", "question id", "item_hash", "narrative"]]


def load_codebook(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def codebook_prompt(codebook: dict[str, Any]) -> str:
    d = codebook["dimensions"]
    lines = [
        "You are a blinded research coder. Your only task is to rate Person B in one vignette state.",
        "You have no access to study outcomes. Do not predict or recommend an allocation.",
        "Use only facts explicitly stated in the vignette. Rate the five dimensions independently.",
        "Return integer scores only within the specified 1-7 scales, plus confidence and a brief audit rationale.",
        "",
        "GENERAL RULES:",
        *[f"- {r}" for r in codebook["general_rules"]],
        "",
        "SCALES:",
    ]
    for key, spec in d.items():
        levels = spec.get("levels", spec.get("anchors", {}))
        lines += [
            f"{key}: {spec['label']}",
            f"Question: {spec['question']}",
            *[f"{i} = {levels[str(i)]}" for i in range(1, 8)],
            "",
        ]
    lines += [
        "CONFIDENCE: 1 = very uncertain, 5 = very confident that the scores follow the codebook.",
        "RATIONALE: <= 60 words. Explain only the textual basis for the ratings. Do not discuss resource allocation.",
    ]
    return "\n".join(lines)


def user_prompt(narrative: str) -> str:
    return (
        "Code the following cumulative vignette state. The ordering/location of this state in the study "
        "has intentionally been withheld.\n\n<VIGNETTE>\n"
        + narrative
        + "\n</VIGNETTE>"
    )


def call_openai(model: str, system: str, prompt: str) -> tuple[ContextRating, str]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install the OpenAI SDK (pip install openai) to use OpenAI coders.") from e
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": "low"},
        text_format=ContextRating,
    )
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed structured output.")
    return parsed, str(getattr(response, "model", model))


def call_anthropic(
    model: str,
    system: str,
    prompt: str,
) -> tuple[ContextRating, str]:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "Install the Anthropic SDK (pip install anthropic) "
            "to use Anthropic coders."
        ) from e

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic()

    msg = client.messages.parse(
        model=model,
        max_tokens=1000,
        system=system,
        messages=[
            {"role": "user", "content": prompt},
        ],
        output_format=ContextRating,
    )

    parsed = getattr(msg, "parsed_output", None)
    if parsed is None:
        raise RuntimeError("Anthropic returned no parsed structured output.")

    return parsed, str(getattr(msg, "model", model))


def call_typesafe(model: str, system: str, prompt: str, codebook: dict[str, Any]) -> tuple[ContextRating, str]:
    """Call TypeSafe AI Jev using five parallel 7-level Score questions.

    Jev is a structured decision model rather than a chat model, so the frozen
    codebook is expressed directly in the five score rubrics. The modal score
    level is used as the discrete 1-7 coder rating; Jev's confidence values are
    retained only as an overall audit field and are not used in the analysis.
    """
    import urllib.error
    import urllib.request

    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set.")

    # user_prompt wraps the narrative in <VIGNETTE> tags; Jev receives just the
    # same blinded text, while each question repeats the relevant coding rule.
    state = prompt
    questions: dict[str, Any] = {}
    for dim, spec in codebook["dimensions"].items():
        levels = spec["levels"]
        shared_rules = " ".join(codebook["general_rules"])
        instructions = (
            f"Blind research coding task. Rate Person B only. {spec['question']} "
            f"Apply these shared coding rules exactly: {shared_rules}"
        )
        questions[dim] = {
            "type": "score",
            "instructions": instructions,
            "criteria": [levels[str(i)] for i in range(1, 8)],
        }

    payload = json.dumps({"model": model, "state": state, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TypeSafe API error {e.code}: {body[:1000]}") from e

    answers = data.get("answers", {})
    ratings: dict[str, int] = {}
    confidences = []
    for dim in codebook["dimensions"]:
        ans = answers.get(dim)
        if not ans:
            raise RuntimeError(f"TypeSafe response missing answer for {dim}")
        probs = ans.get("probabilities", {})
        if probs:
            # Score indices are 0..6. Use the modal level, not the continuous
            # probability-weighted score, to preserve the shared integer rubric.
            best = max(probs.items(), key=lambda kv: float(kv[1]))[0]
            ratings[dim] = int(best) + 1
        else:
            ratings[dim] = int(round(float(ans["score"]))) + 1
        confidences.append(float(ans.get("confidence", 0.5)))

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.5
    confidence_1_5 = int(min(5, max(1, round(1 + 4 * mean_conf))))
    rating = ContextRating(
        **ratings,
        confidence=confidence_1_5,
        rationale="TypeSafe Jev structured score outputs; Jev does not generate a free-text coding rationale.",
    )
    return rating, str(data.get("model", model))


def call_coder(provider: str, model: str, system: str, prompt: str, codebook: dict[str, Any]) -> tuple[ContextRating, str]:
    if provider == "openai":
        return call_openai(model, system, prompt)
    if provider == "anthropic":
        return call_anthropic(model, system, prompt)
    if provider == "typesafe":
        return call_typesafe(model, system, prompt, codebook)
    raise ValueError(f"Unsupported provider {provider!r}. Use openai, anthropic, or typesafe.")


def split_coder(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Coder {spec!r} must use provider:model format.")
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in {"openai", "anthropic", "typesafe"} or not model:
        raise ValueError(f"Invalid coder specification: {spec!r}")
    return provider, model


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["coder_id", "scenario", "level"]).to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    # Load secrets locally without ever writing them to outputs. Shell/CI
    # environment variables take precedence over values in .env.
    env_loaded = False
    if args.env_file.exists():
        env_loaded = bool(load_dotenv(dotenv_path=args.env_file, override=False))
    else:
        print(f"Note: {args.env_file} not found; using already-set environment variables only.")

    states = load_states(args.input_dir, args.pattern)
    codebook, codebook_hash = load_codebook(args.codebook)
    system = codebook_prompt(codebook)

    # This local manifest retains scenario/level solely so returned ratings can be
    # merged back to the CRAV study. It is NEVER included in an API request.
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(args.manifest, index=False)

    coder_specs = [split_coder(c) for c in args.coders]

    # Precompute and archive each coder's deterministic random item order. The
    # APIs receive only the narrative text for the current item, not these labels.
    order_rows = []
    coder_orders: dict[str, list[int]] = {}
    for provider, model in coder_specs:
        coder_id = f"{provider}:{model}"
        order = list(states.index)
        random.Random(stable_seed(args.seed, coder_id)).shuffle(order)
        coder_orders[coder_id] = order
        for position, idx in enumerate(order, 1):
            state = states.loc[idx]
            order_rows.append({
                "coder_id": coder_id,
                "presentation_position": position,
                "item_hash": state["item_hash"],
                "scenario_local_only": int(state["scenario"]),
                "level_local_only": int(state["level"]),
            })
    args.order_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(order_rows).to_csv(args.order_manifest, index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "codebook_file": args.codebook.name,
        "codebook_version": codebook.get("version"),
        "codebook_sha256": codebook_hash,
        "n_states": int(len(states)),
        "input_source": f"{args.input_dir}/{args.pattern}",
        "env_file": str(args.env_file),
        "env_file_loaded": env_loaded,
        "secrets_policy": "API key values are read at runtime only and are never written to manifests, metadata, or coding outputs.",
        "blindness": "Coding APIs receive only the frozen codebook/rubric and one cumulative narrative state; no scenario/level labels, allocation outcomes, transition effects, or existing theory-map classifications are sent.",
        "randomisation": "Each coder receives all 50 states once in a deterministic coder-specific random order derived from the recorded seed.",
        "coders": [{"provider": p, "model": m, "coder_id": f"{p}:{m}"} for p, m in coder_specs],
        "seed_used_only_for_item_order": args.seed,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.prepare_only:
        print(f"Prepared blinded state manifest: {args.manifest}")
        print(f"Prepared coder-order manifest: {args.order_manifest}")
        print(f"Codebook SHA-256: {codebook_hash}")
        return

    rows: list[dict[str, Any]] = []
    if args.output.exists() and not args.overwrite:
        rows = pd.read_csv(args.output).to_dict("records")
        existing = {(str(r["coder_id"]), str(r["item_hash"])) for r in rows}
    else:
        existing = set()

    for provider, model in coder_specs:
        coder_id = f"{provider}:{model}"
        order = coder_orders[coder_id]
        print(f"\n{coder_id}: {len(order)} blinded states")

        for pos, idx in enumerate(order, 1):
            state = states.loc[idx]
            key = (coder_id, state["item_hash"])
            if key in existing:
                continue

            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    rating, resolved_model = call_coder(
                        provider, model, system, user_prompt(state["narrative"]), codebook
                    )
                    break
                except Exception as e:
                    last_error = e
                    if attempt == args.max_retries:
                        raise
                    time.sleep(min(30, 2 ** attempt))
            else:  # pragma: no cover
                raise RuntimeError(last_error)

            record = {
                "coder_id": coder_id,
                "provider": provider,
                "model": model,
                "resolved_model": resolved_model,
                "codebook_version": codebook.get("version"),
                "codebook_sha256": codebook_hash,
                "item_hash": state["item_hash"],
                "scenario": int(state["scenario"]),
                "level": int(state["level"]),
                "narrative": state["narrative"],
                **rating.model_dump(),
                "coded_utc": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(record)
            existing.add(key)
            save_rows(rows, args.output)
            print(f"  {pos:02d}/50 complete", end="\r", flush=True)
        print("  complete" + " " * 20)

    out = pd.DataFrame(rows)
    expected = len(coder_specs) * 50
    if len(out) != expected:
        raise ValueError(f"Expected {expected} coder-state rows, found {len(out)}")
    for col in [
        "responsibility_controllability", "need_vulnerability", "external_constraint_coercion",
        "mitigating_circumstances_motive", "corrective_prosocial_effort",
    ]:
        if not out[col].between(1, 7).all():
            raise ValueError(f"Out-of-range ratings detected in {col}")

    print(f"\nSaved {len(out)} blinded ratings to {args.output}")
    print(f"Codebook SHA-256: {codebook_hash}")
    print("Run crav_analysis.py next; it will automatically analyse the coding file if present.")


if __name__ == "__main__":
    main()
