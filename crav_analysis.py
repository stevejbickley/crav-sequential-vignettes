#!/usr/bin/env python3
"""
CRAV analysis for:
"Context-Sensitive Resource Allocation:
 Evidence from Sequential Vignettes Using Artificial Agents"

This script:
1. Loads the 10 archived SurveyLM CRAV CSV exports.
2. Parses Person A/B dollar allocations from free-text answers.
3. Validates the $1,000 allocation constraint and balanced repeated design.
4. Builds a clean long panel (agent x scenario x level).
5. Summarises allocation trajectories and stage-to-stage reallocations.
6. Tests each of the 40 within-agent transitions (paired t-test + Wilcoxon),
   with Benjamini-Hochberg false-discovery-rate correction.
7. Runs a two-way repeated-measures ANOVA (scenario x contextual level).
8. Fits descriptive model specifications comparing R-squared across scenario,
   contextual level, model-instance identity, and temperature.
9. Produces Main Figure 1 and Table 1 and Supplementary Figures S1-S2 and
   Tables S1-S8.
10. Writes Supplementary Data S1-S2, validation outputs, analysis metadata,
    recorded environment versions, and a manuscript numerical cross-check.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


TOTAL_BUDGET = 1000.0
EXPECTED_SCENARIOS = 10
EXPECTED_LEVELS = 5
EXPECTED_AGENTS = 50
EXPECTED_ROWS = EXPECTED_SCENARIOS * EXPECTED_LEVELS * EXPECTED_AGENTS
RANDOM_SEED = 20260901

PLOT_TITLES: Dict[int, str] = {
    1: "Mobility impairment / criminal activity",
    2: "Pregnancy loss / abortion",
    3: "Trauma / insurgency",
    4: "Disability / drunk driving",
    5: "Homelessness / former athlete",
    6: "Unemployment / workplace theft",
    7: "Emphysema / medical need",
    8: "Refugee / sick child",
    9: "Startup failure / recession",
    10: "Ex-convict / rehabilitation",
}

SCENARIO_NAMES: Dict[int, str] = {
    1: "Mobility impairment / criminal activity",
    2: "Pregnancy loss / abortion",
    3: "Communication trauma / insurgency",
    4: "Disability / drunk-driving accident",
    5: "Homelessness / former athlete",
    6: "Unemployment / workplace theft",
    7: "Emphysema / smoking / medical need",
    8: "Refugee / sick child / document forgery",
    9: "Startup failure / recession / social enterprise",
    10: "Ex-convict / family need / rehabilitation",
}

# Short labels are used only for plotting/tables; the full vignette wording is
# extracted directly from the source data and saved separately.
STAGE_CUES: Dict[int, Dict[int, str]] = {
    1: {
        1: "Mobility issues",
        2: "Criminal activity",
        3: "Forced gang membership",
        4: "Illegal gambling debts",
        5: "Gambling for child's cancer care",
    },
    2: {
        1: "A pregnant; B no added condition",
        2: "Pregnancy loss",
        3: "Abortion",
        4: "Life-saving medical abortion",
        5: "Incest complications",
    },
    3: {
        1: "Communication/speech problems",
        2: "Trauma",
        3: "Trauma as insurgent",
        4: "Family killed in drone bombing",
        5: "Family in insurgency leadership",
    },
    4: {
        1: "Permanent disability after crash",
        2: "Drunk driving",
        3: "Drinking linked to bereavement",
        4: "Friend died in war",
        5: "B survived same ambush",
    },
    5: {
        1: "Homelessness",
        2: "Former professional athlete",
        3: "Career-ending injury",
        4: "Painkiller addiction",
        5: "Seeking rehab / fatherhood",
    },
    6: {
        1: "Unemployed",
        2: "Theft from employer",
        3: "Cannot re-enter field",
        4: "Working in fast food",
        5: "Sole family breadwinner",
    },
    7: {
        1: "Emphysema",
        2: "Years of smoking",
        3: "No insurance coverage",
        4: "Cannot pay medical bills",
        5: "Treatment versus rent",
    },
    8: {
        1: "Newly arrived refugee",
        2: "Sick child in shelter",
        3: "Forged documents for urgent care",
        4: "Faces deportation",
        5: "Spouse missing in conflict zone",
    },
    9: {
        1: "Startup at risk",
        2: "Investor lost in recession",
        3: "High-interest loan",
        4: "Default / personal bankruptcy",
        5: "Pivot to youth social enterprise",
    },
    10: {
        1: "Recently released ex-convict",
        2: "Offence to feed family",
        3: "Family left / transitional housing",
        4: "Pressure to re-offend",
        5: "Rehabilitation / volunteering",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse CRAV SurveyLM exports.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_data"),
        help="Directory containing the 10 completed_survey_data CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Package/output root. Creates data/, tables/, figures/, and checks/ subdirectories.",
    )
    parser.add_argument(
        "--pattern",
        default="crav_vignette_*_raw.csv",
        help="Glob pattern for the 10 archived SurveyLM CRAV CSV files.",
    )
    parser.add_argument(
        "--allow-unexpected-counts",
        action="store_true",
        help="Continue if the number of files/agents/rows differs from the supplied study.",
    )
    return parser.parse_args()


def make_output_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "data": root / "data",
        "tables": root / "tables",
        "figures": root / "figures",
        "checks": root / "checks",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_data(input_dir: Path, pattern: str) -> Tuple[pd.DataFrame, list[Path]]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {input_dir.resolve()}")

    frames = []
    for f in files:
        x = pd.read_csv(f)
        x["source_file"] = f.name
        frames.append(x)
    return pd.concat(frames, ignore_index=True), files


def parse_scenario_level(question_id: str) -> Tuple[int, int]:
    """Parse Scenario_1_V2_Level_3 or Scenario_6_Level_4 -> (scenario, level)."""
    m = re.search(r"Scenario_(\d+)(?:_V\d+)?_Level_(\d+)", str(question_id))
    if not m:
        raise ValueError(f"Cannot parse scenario/level from question id: {question_id!r}")
    return int(m.group(1)), int(m.group(2))


def parse_allocations(answer: str) -> Tuple[float, float, str]:
    """
    Parse Person A and Person B allocations from the observed answer styles.

    Supported examples include:
      Person A: $400, Person B: $600
      Allocate $600 to Person B and $400 to Person A ...
      $400 to Person A, $600 to Person B
      Allocate $500 to both Person A and Person B equally.

    If exactly one person's amount is identified, the other is inferred from
    the fixed $1,000 budget. The downstream validation step rejects any row
    that does not sum to $1,000 or lies outside [0, 1000].
    """
    text = " ".join(str(answer).replace(",", "").split())

    # Explicit equal-to-both construction.
    m = re.search(
        r"(?:allocate|give|I would allocate)?\s*(?:USD\s*)?\$?\s*"
        r"(\d+(?:\.\d+)?)\s+(?:USD\s+)?to\s+both\s+Person\s+A\s+and\s+Person\s+B",
        text,
        flags=re.I,
    )
    if m:
        value = float(m.group(1))
        return value, value, "explicit_both"

    out: Dict[str, float] = {}
    for person in ("A", "B"):
        patterns = [
            # "$600 to Person A"
            rf"(?:USD\s*)?\$\s*(\d+(?:\.\d+)?)\s*(?:USD)?\s*(?:to|for)\s+Person\s*{person}\b",
            # "600 USD to Person A"
            rf"(?<!\d)(\d+(?:\.\d+)?)\s*(?:USD)\s*(?:to|for)\s+Person\s*{person}\b",
            # "Person A: $600"
            rf"Person\s*{person}\s*[:=\-]\s*(?:USD\s*)?\$?\s*(\d+(?:\.\d+)?)\b",
            # "allocate 600 to Person A"
            rf"(?:allocate|give|giving)\s+(?:USD\s*)?\$?\s*(\d+(?:\.\d+)?)\s*"
            rf"(?:USD)?\s*(?:to|for)\s+Person\s*{person}\b",
        ]
        for pat in patterns:
            match = re.search(pat, text, flags=re.I)
            if match:
                out[person] = float(match.group(1))
                break

    a = out.get("A", np.nan)
    b = out.get("B", np.nan)
    method = "explicit_A_and_B"

    if np.isnan(a) and np.isfinite(b) and 0 <= b <= TOTAL_BUDGET:
        a = TOTAL_BUDGET - b
        method = "infer_A_from_B"
    if np.isnan(b) and np.isfinite(a) and 0 <= a <= TOTAL_BUDGET:
        b = TOTAL_BUDGET - a
        method = "infer_B_from_A"

    return a, b, method


def prepare_panel(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "agent",
        "question",
        "question id",
        "answer",
        "model",
        "temperature",
        "Persona",
        "created",
        "simulation id",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = raw.copy()
    parsed = out["question id"].map(parse_scenario_level)
    out[["scenario", "level"]] = pd.DataFrame(parsed.tolist(), index=out.index)

    allocations = out["answer"].map(parse_allocations)
    out[["allocation_A", "allocation_B", "parse_method"]] = pd.DataFrame(
        allocations.tolist(), index=out.index
    )

    out["allocation_B_share"] = out["allocation_B"] / TOTAL_BUDGET
    out["allocation_B_pct"] = 100.0 * out["allocation_B_share"]
    out["scenario_name"] = out["scenario"].map(SCENARIO_NAMES)
    out["stage_cue"] = [
        STAGE_CUES[int(s)][int(l)] for s, l in zip(out["scenario"], out["level"])
    ]
    out["created"] = pd.to_datetime(out["created"], errors="coerce", utc=True)

    return out


def validate_panel(df: pd.DataFrame, files: Iterable[Path], allow_unexpected: bool) -> pd.DataFrame:
    checks = []

    def add_check(name: str, passed: bool, value, expected=""):
        checks.append(
            {"check": name, "passed": bool(passed), "observed": value, "expected": expected}
        )

    # Parse/budget checks.
    parsed_ok = df[["allocation_A", "allocation_B"]].notna().all(axis=1)
    sums = df["allocation_A"] + df["allocation_B"]
    budget_ok = np.isclose(sums, TOTAL_BUDGET, atol=1e-6)
    bounds_ok = (
        df["allocation_A"].between(0, TOTAL_BUDGET)
        & df["allocation_B"].between(0, TOTAL_BUDGET)
    )
    duplicate_count = int(df.duplicated(["agent", "scenario", "level"]).sum())

    add_check("all allocations parsed", parsed_ok.all(), int(parsed_ok.sum()), len(df))
    add_check("all allocations sum to $1000", budget_ok.all(), int(budget_ok.sum()), len(df))
    add_check("all allocations within [0,1000]", bounds_ok.all(), int(bounds_ok.sum()), len(df))
    add_check("no duplicate agent-scenario-level rows", duplicate_count == 0, duplicate_count, 0)
    add_check("number of input CSV files", len(list(files)) == 10, len(list(files)), 10)
    add_check("number of rows", len(df) == EXPECTED_ROWS, len(df), EXPECTED_ROWS)
    add_check("number of agents", df["agent"].nunique() == EXPECTED_AGENTS, df["agent"].nunique(), EXPECTED_AGENTS)
    add_check("number of scenarios", df["scenario"].nunique() == EXPECTED_SCENARIOS, df["scenario"].nunique(), EXPECTED_SCENARIOS)
    add_check("number of levels", df["level"].nunique() == EXPECTED_LEVELS, df["level"].nunique(), EXPECTED_LEVELS)

    # Every agent should have all 50 scenario-level cells.
    per_agent = df.groupby("agent").size()
    add_check(
        "balanced 50 observations per agent",
        (per_agent == EXPECTED_SCENARIOS * EXPECTED_LEVELS).all(),
        f"min={per_agent.min()}, max={per_agent.max()}",
        EXPECTED_SCENARIOS * EXPECTED_LEVELS,
    )

    # Temperature is expected to be fixed within agent in supplied data.
    temp_nunique = df.groupby("agent")["temperature"].nunique()
    add_check(
        "temperature fixed within agent",
        (temp_nunique == 1).all(),
        int((temp_nunique == 1).sum()),
        df["agent"].nunique(),
    )

    qc = pd.DataFrame(checks)

    hard_fail = not (
        parsed_ok.all()
        and budget_ok.all()
        and bounds_ok.all()
        and duplicate_count == 0
    )
    if hard_fail:
        bad = df.loc[~(parsed_ok & budget_ok & bounds_ok), [
            "agent", "question id", "answer", "allocation_A", "allocation_B"
        ]]
        raise ValueError(
            "Allocation parsing/validation failed. Example problematic rows:\n"
            + bad.head(20).to_string(index=False)
        )

    if not allow_unexpected:
        structural_checks = qc.loc[
            qc["check"].isin(
                [
                    "number of input CSV files",
                    "number of rows",
                    "number of agents",
                    "number of scenarios",
                    "number of levels",
                    "balanced 50 observations per agent",
                ]
            )
        ]
        if not structural_checks["passed"].all():
            raise ValueError(
                "Study structure differs from the expected 10 x 5 x 50 design. "
                "Use --allow-unexpected-counts only if this is intentional.\n"
                + structural_checks.to_string(index=False)
            )

    return qc


def mean_ci_t(x: pd.Series, confidence: float = 0.95) -> Tuple[float, float, float, float, int]:
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    if n <= 1 or np.isclose(sd, 0.0):
        return mean, sd, mean, mean, n
    se = sd / math.sqrt(n)
    crit = stats.t.ppf(0.5 + confidence / 2.0, df=n - 1)
    return mean, sd, mean - crit * se, mean + crit * se, n


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, level), g in df.groupby(["scenario", "level"], sort=True):
        mean, sd, lo, hi, n = mean_ci_t(g["allocation_B_pct"])
        se = sd / math.sqrt(n) if n > 1 else np.nan
        rows.append(
            {
                "scenario": scenario,
                "scenario_name": SCENARIO_NAMES[int(scenario)],
                "level": level,
                "new_information": STAGE_CUES[int(scenario)][int(level)],
                "n_model_instances": n,
                "mean_B_pct": mean,
                "sd_B_pct": sd,
                "se_B_pct": se,
                "median_B_pct": float(g["allocation_B_pct"].median()),
                "ci95_low": lo,
                "ci95_high": hi,
                "mean_B_usd": mean * 10.0,
            }
        )
    return pd.DataFrame(rows)

def overall_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over scenarios within model instance before inference."""
    agent_level = (
        df.groupby(["agent", "level"], as_index=False)["allocation_B_pct"]
        .mean()
        .rename(columns={"allocation_B_pct": "agent_mean_B_pct"})
    )
    wide = agent_level.pivot(index="agent", columns="level", values="agent_mean_B_pct")
    rows = []
    for level in range(1, EXPECTED_LEVELS + 1):
        g = wide[level].dropna()
        mean, sd, lo, hi, n = mean_ci_t(g)
        se = sd / math.sqrt(n) if n > 1 else np.nan
        row = {
            "level": level,
            "n_model_instances": n,
            "mean_B_pct": mean,
            "sd_agent_mean_B_pct": sd,
            "se_agent_mean_B_pct": se,
            "ci95_low": lo,
            "ci95_high": hi,
            "mean_delta_from_prior_pp": np.nan,
            "delta_ci95_low": np.nan,
            "delta_ci95_high": np.nan,
            "paired_t": np.nan,
            "paired_t_p": np.nan,
        }
        if level > 1:
            delta = wide[level] - wide[level - 1]
            dmean, dsd, dlo, dhi, dn = mean_ci_t(delta)
            test = stats.ttest_rel(wide[level], wide[level - 1], nan_policy="omit")
            row.update({
                "mean_delta_from_prior_pp": dmean,
                "delta_ci95_low": dlo,
                "delta_ci95_high": dhi,
                "paired_t": float(test.statistic),
                "paired_t_p": float(test.pvalue),
            })
        rows.append(row)
    return pd.DataFrame(rows)

def transition_tests(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute 40 within-agent level-to-level changes and paired tests."""
    wide = df.pivot(index="agent", columns=["scenario", "level"], values="allocation_B_pct")
    rows = []
    delta_long = []

    for scenario in range(1, EXPECTED_SCENARIOS + 1):
        for to_level in range(2, EXPECTED_LEVELS + 1):
            from_level = to_level - 1
            before = wide[(scenario, from_level)]
            after = wide[(scenario, to_level)]
            delta = after - before

            mean, sd, lo, hi, n = mean_ci_t(delta)

            if np.isclose(delta.std(ddof=1), 0.0):
                # All paired differences identical. A standard t statistic is
                # undefined/infinite; record the limiting inference explicitly.
                t_stat = np.sign(mean) * np.inf if not np.isclose(mean, 0.0) else 0.0
                t_p = 0.0 if not np.isclose(mean, 0.0) else 1.0
            else:
                t_result = stats.ttest_rel(after, before, nan_policy="omit")
                t_stat, t_p = float(t_result.statistic), float(t_result.pvalue)

            try:
                w_result = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
                w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
            except ValueError:
                # Happens only if every paired difference is zero.
                w_stat, w_p = np.nan, 1.0

            cue = STAGE_CUES[scenario][to_level]
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_name": SCENARIO_NAMES[scenario],
                    "from_level": from_level,
                    "to_level": to_level,
                    "transition": f"L{from_level}→L{to_level}",
                    "new_information": cue,
                    "n_agents": n,
                    "mean_delta_pp": mean,
                    "sd_delta_pp": sd,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "median_delta_pp": float(delta.median()),
                    "prop_increase": float((delta > 0).mean()),
                    "prop_decrease": float((delta < 0).mean()),
                    "prop_no_change": float((delta == 0).mean()),
                    "paired_t": t_stat,
                    "paired_t_p": t_p,
                    "wilcoxon_W": w_stat,
                    "wilcoxon_p": w_p,
                }
            )

            for agent_id, value in delta.items():
                delta_long.append(
                    {
                        "agent": agent_id,
                        "scenario": scenario,
                        "from_level": from_level,
                        "to_level": to_level,
                        "transition": f"L{from_level}→L{to_level}",
                        "new_information": cue,
                        "delta_B_pp": float(value),
                    }
                )

    results = pd.DataFrame(rows)
    results["paired_t_q_BH"] = multipletests(results["paired_t_p"], method="fdr_bh")[1]
    results["wilcoxon_q_BH"] = multipletests(results["wilcoxon_p"], method="fdr_bh")[1]
    results["significant_FDR_05"] = results["paired_t_q_BH"] < 0.05
    results["direction"] = np.select(
        [results["mean_delta_pp"] < 0, results["mean_delta_pp"] > 0],
        ["decrease", "increase"],
        default="no change",
    )

    return results, pd.DataFrame(delta_long)


def scenario_summary(cells: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in range(1, EXPECTED_SCENARIOS + 1):
        c = cells[cells["scenario"] == scenario].sort_values("level")
        t = transitions[transitions["scenario"] == scenario].copy()
        level_means = c["mean_B_pct"].to_numpy()
        diffs = np.diff(level_means)

        if np.allclose(diffs, 0):
            trajectory = "constant"
        elif np.all(diffs >= 0):
            trajectory = "monotonic increase"
        elif np.all(diffs <= 0):
            trajectory = "monotonic decrease"
        else:
            trajectory = "non-monotonic"

        negative = t[t["mean_delta_pp"] < 0]
        positive = t[t["mean_delta_pp"] > 0]

        if negative.empty:
            largest_fall = "—"
            largest_fall_pp = np.nan
        else:
            r = negative.loc[negative["mean_delta_pp"].idxmin()]
            largest_fall = f"{r['transition']}: {r['new_information']} ({r['mean_delta_pp']:.1f} pp)"
            largest_fall_pp = float(r["mean_delta_pp"])

        if positive.empty:
            largest_rise = "—"
            largest_rise_pp = np.nan
        else:
            r = positive.loc[positive["mean_delta_pp"].idxmax()]
            largest_rise = f"{r['transition']}: {r['new_information']} (+{r['mean_delta_pp']:.1f} pp)"
            largest_rise_pp = float(r["mean_delta_pp"])

        rows.append(
            {
                "scenario": scenario,
                "scenario_name": SCENARIO_NAMES[scenario],
                "L1_mean_B_pct": float(level_means[0]),
                "L5_mean_B_pct": float(level_means[-1]),
                "net_change_pp": float(level_means[-1] - level_means[0]),
                "negative_transitions": int((t["mean_delta_pp"] < 0).sum()),
                "trajectory": trajectory,
                "largest_fall": largest_fall,
                "largest_fall_pp": largest_fall_pp,
                "largest_rise": largest_rise,
                "largest_rise_pp": largest_rise_pp,
            }
        )
    return pd.DataFrame(rows)


def repeated_measures_anova(df: pd.DataFrame) -> pd.DataFrame:
    aov = AnovaRM(
        data=df,
        depvar="allocation_B_pct",
        subject="agent",
        within=["scenario", "level"],
    ).fit()
    raw = aov.anova_table.reset_index().rename(columns={"index": "effect"})
    return pd.DataFrame({
        "effect": raw["effect"],
        "df_num": raw["Num DF"].astype(float),
        "df_den": raw["Den DF"].astype(float),
        "F": raw["F Value"].astype(float),
        "p": raw["Pr > F"].astype(float),
    })

def model_fit_table(df: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("Temperature only", "allocation_B_pct ~ C(temperature)"),
        ("Model-instance identity only", "allocation_B_pct ~ C(agent)"),
        ("Scenario only", "allocation_B_pct ~ C(scenario)"),
        ("Contextual level only", "allocation_B_pct ~ C(level)"),
        ("Scenario + level", "allocation_B_pct ~ C(scenario) + C(level)"),
        ("Scenario × level", "allocation_B_pct ~ C(scenario) * C(level)"),
        ("Scenario × level + temperature", "allocation_B_pct ~ C(scenario) * C(level) + C(temperature)"),
    ]
    rows = []
    for name, formula in specifications:
        fit = smf.ols(formula, data=df).fit()
        rows.append({
            "model": name,
            "formula": formula,
            "R2": float(fit.rsquared),
            "adjusted_R2": float(fit.rsquared_adj),
            "AIC": float(fit.aic),
            "BIC": float(fit.bic),
            "n_observations": int(fit.nobs),
            "n_parameters": int(fit.df_model + 1),
            "interpretation": "Descriptive model fit; not a causal variance decomposition.",
        })
    return pd.DataFrame(rows)

def temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    agent_temp = df[["agent", "temperature"]].drop_duplicates()
    agent_mean = df.groupby(["agent", "temperature"], as_index=False)["allocation_B_pct"].mean()
    rows = []
    for temp, g in agent_mean.groupby("temperature", sort=True):
        mean, sd, lo, hi, n_agents = mean_ci_t(g["allocation_B_pct"])
        rows.append({
            "temperature": float(temp),
            "n_model_instances": int(n_agents),
            "n_allocation_observations": int((df["temperature"] == temp).sum()),
            "mean_instance_B_pct": mean,
            "sd_instance_B_pct": sd,
            "ci95_low": lo,
            "ci95_high": hi,
        })
    return pd.DataFrame(rows)

def vignette_table(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df[["scenario", "scenario_name", "level", "question id", "question", "answer instruction"]]
        .drop_duplicates(["scenario", "level"])
        .sort_values(["scenario", "level"])
        .reset_index(drop=True)
    )
    base["transition"] = base["level"].map(lambda l: "Initial presentation" if int(l) == 1 else f"L{int(l)-1}→L{int(l)}")
    base["new_information"] = [STAGE_CUES[int(s)][int(l)] for s, l in zip(base["scenario"], base["level"])]
    return base[[
        "scenario", "scenario_name", "level", "transition", "new_information",
        "question id", "question", "answer instruction"
    ]]

def save_table(df: pd.DataFrame, stem: Path, index: bool = False) -> None:
    """Save CSV plus a readable Markdown version."""
    df.to_csv(stem.with_suffix(".csv"), index=index)
    try:
        stem.with_suffix(".md").write_text(df.to_markdown(index=index), encoding="utf-8")
    except Exception:
        # to_markdown requires tabulate; CSV remains canonical if unavailable.
        pass


def figure1_trajectories(cells: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    """Main figure: ten small-multiple mean trajectories with 95% CIs."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.8), sharex=True, sharey=True)
    axes = axes.ravel()

    y_min = max(0, math.floor((cells["ci95_low"].min() - 5) / 5) * 5)
    y_max = min(100, math.ceil((cells["ci95_high"].max() + 5) / 5) * 5)

    for scenario, ax in zip(range(1, 11), axes):
        c = cells[cells["scenario"] == scenario].sort_values("level")
        x = c["level"].to_numpy()
        y = c["mean_B_pct"].to_numpy()
        yerr = np.vstack([y - c["ci95_low"].to_numpy(), c["ci95_high"].to_numpy() - y])

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.6)
        ax.axhline(50, linestyle="--", linewidth=0.9)
        ax.set_title(f"S{scenario}. {PLOT_TITLES[scenario]}", fontsize=9)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(["L1", "L2", "L3", "L4", "L5"])
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.18)

        # Point labels make the figure interpretable in grayscale/print.
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)

    for ax in axes[5:]:
        ax.set_xlabel("Sequential contextual level")
    for ax in axes[::5]:
        ax.set_ylabel("Allocation to Person B (%)")

    fig.suptitle(
        "CRAV allocation trajectories across sequential contextual revelations",
        fontsize=13,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Points are means across 50 repeated GPT-4o agent IDs; error bars are 95% t confidence intervals. "
        "Dashed line denotes an equal 50/50 split.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.96])
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def figure_s1_transitions(transitions: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    """Supplement: all 40 paired mean changes with 95% CIs."""
    t = transitions.copy().sort_values(["scenario", "to_level"], ascending=[False, False]).reset_index(drop=True)
    labels = [
        f"S{int(r.scenario)} {r.transition}: {r.new_information}"
        for r in t.itertuples(index=False)
    ]
    y = np.arange(len(t))
    means = t["mean_delta_pp"].to_numpy()
    xerr = np.vstack([
        means - t["ci95_low"].to_numpy(),
        t["ci95_high"].to_numpy() - means,
    ])

    fig, ax = plt.subplots(figsize=(11, 14))
    ax.errorbar(means, y, xerr=xerr, fmt="o", capsize=2, linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Change in allocation to Person B (percentage points)")
    ax.set_title("Stage-to-stage contextual reallocations")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def figure_s2_overall(overall: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = overall["level"].to_numpy()
    y = overall["mean_B_pct"].to_numpy()
    yerr = np.vstack([y - overall["ci95_low"].to_numpy(), overall["ci95_high"].to_numpy() - y])
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=1.6)
    ax.axhline(50, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Sequential contextual level")
    ax.set_ylabel("Mean allocation to Person B (%)")
    ax.set_title("Average allocation across the ten CRAV scenarios")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def manuscript_summary_text(
    df: pd.DataFrame,
    overall: pd.DataFrame,
    transitions: pd.DataFrame,
    scenarios: pd.DataFrame,
    model_fits: pd.DataFrame,
    anova: pd.DataFrame,
) -> str:
    agent_level = df.groupby(["agent", "level"])["allocation_B_pct"].mean().unstack()
    l1_l5_delta = agent_level[5] - agent_level[1]
    d_mean, d_sd, d_lo, d_hi, d_n = mean_ci_t(l1_l5_delta)
    d_test = stats.ttest_rel(agent_level[5], agent_level[1])

    n_negative = int((transitions["mean_delta_pp"] < 0).sum())
    n_positive = int((transitions["mean_delta_pp"] > 0).sum())
    n_sig = int((transitions["paired_t_q_BH"] < 0.05).sum())
    n_nonmono = int((scenarios["trajectory"] == "non-monotonic").sum())

    def tr(s, f, to):
        return transitions.query("scenario == @s and from_level == @f and to_level == @to").iloc[0]

    selected = [
        ("S1 criminal activity", tr(1,1,2)),
        ("S2 abortion after pregnancy loss", tr(2,2,3)),
        ("S3 family insurgency leadership", tr(3,4,5)),
        ("S4 drunk driving", tr(4,1,2)),
        ("S4 bereavement-linked drinking", tr(4,2,3)),
        ("S6 theft from employer", tr(6,1,2)),
        ("S6 sole breadwinner", tr(6,4,5)),
        ("S7 smoking cause", tr(7,1,2)),
        ("S8 forged documents for urgent care", tr(8,2,3)),
    ]

    fit_lookup = model_fits.set_index("model")
    lines = [
        "CRAV MANUSCRIPT NUMERICAL CROSS-CHECK",
        "=" * 55,
        f"Observations: {len(df):,}; model instances: {df['agent'].nunique()}; scenarios: 10; levels: 5.",
        f"Model(s): {', '.join(map(str, df['model'].dropna().unique()))}.",
        f"Collection window: {df['created'].min()} to {df['created'].max()} (UTC).",
        "",
        "Overall contextual progression",
        f"Level 1 mean B allocation: {overall.loc[overall.level == 1, 'mean_B_pct'].iloc[0]:.3f}%.",
        f"Level 5 mean B allocation: {overall.loc[overall.level == 5, 'mean_B_pct'].iloc[0]:.3f}%.",
        f"Model-instance mean L5-L1 change: {d_mean:.3f} pp (95% CI {d_lo:.3f}, {d_hi:.3f}); paired t({d_n-1})={d_test.statistic:.3f}, p={d_test.pvalue:.3e}.",
        "",
        "Transition structure",
        f"{n_sig}/40 paired transitions significant at BH-FDR q<.05; {n_positive} positive and {n_negative} negative mean changes.",
        f"Non-monotonic scenario-mean trajectories: {n_nonmono}/10.",
        "",
        "Selected contextual shifts (percentage points to Person B)",
    ]
    for label, r in selected:
        lines.append(f"- {label}: {r.mean_delta_pp:+.2f} pp (95% CI {r.ci95_low:.2f}, {r.ci95_high:.2f}; BH q={r.paired_t_q_BH:.3e}).")

    lines += ["", "Descriptive R-squared"]
    for name in [
        "Temperature only",
        "Model-instance identity only",
        "Scenario only",
        "Contextual level only",
        "Scenario + level",
        "Scenario × level",
        "Scenario × level + temperature",
    ]:
        lines.append(f"- {name}: R2={fit_lookup.loc[name, 'R2']:.6f}; adjusted R2={fit_lookup.loc[name, 'adjusted_R2']:.6f}")

    lines += ["", "Two-way repeated-measures ANOVA"]
    for r in anova.itertuples(index=False):
        lines.append(f"- {r.effect}: F({r.df_num:.0f},{r.df_den:.0f})={r.F:.3f}, p={r.p:.3e}.")

    return "\n".join(lines) + "\n"

def write_metadata(df: pd.DataFrame, files: list[Path], out_path: Path) -> None:
    metadata = {
        "input_files": [f.name for f in files],
        "n_rows": int(len(df)),
        "n_agents": int(df["agent"].nunique()),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_levels": int(df["level"].nunique()),
        "models": [str(x) for x in df["model"].dropna().unique()],
        "personas": [str(x) for x in df["Persona"].dropna().unique()],
        "temperatures": sorted(float(x) for x in df["temperature"].dropna().unique()),
        "created_min_utc": str(df["created"].min()),
        "created_max_utc": str(df["created"].max()),
        "total_budget_usd": TOTAL_BUDGET,
        "analysis_random_seed": RANDOM_SEED,
    }
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")



def write_environment_versions(out_path: Path) -> None:
    import platform
    import matplotlib
    import scipy
    import statsmodels
    lines = [
        f"Python {platform.python_version()}",
        f"pandas {pd.__version__}",
        f"numpy {np.__version__}",
        f"scipy {scipy.__version__}",
        f"statsmodels {statsmodels.__version__}",
        f"matplotlib {matplotlib.__version__}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    dirs = make_output_dirs(args.output_dir)

    raw, files = load_data(args.input_dir, args.pattern)
    df = prepare_panel(raw)
    qc = validate_panel(df, files, args.allow_unexpected_counts)

    clean_cols = [
        "source_file", "simulation id", "created", "agent", "model", "temperature",
        "Persona", "scenario", "scenario_name", "level", "stage_cue", "question id",
        "question", "answer instruction", "answer", "allocation_A", "allocation_B",
        "allocation_B_share", "allocation_B_pct", "parse_method",
    ]
    df[clean_cols].sort_values(["agent", "scenario", "level"]).to_csv(
        dirs["data"] / "supplementary_data_s1_crav_long_clean.csv", index=False
    )

    cells = cell_summary(df)
    overall = overall_level_summary(df)
    transitions, delta_long = transition_tests(df)
    scenarios = scenario_summary(cells, transitions)
    anova = repeated_measures_anova(df)
    fits = model_fit_table(df)
    temps = temperature_summary(df)
    vignettes = vignette_table(df)

    delta_long.to_csv(dirs["data"] / "supplementary_data_s2_transition_deltas.csv", index=False)

    main_table = scenarios[[
        "scenario", "scenario_name", "L1_mean_B_pct", "L5_mean_B_pct",
        "net_change_pp", "trajectory", "largest_fall", "largest_rise",
    ]].copy()

    save_table(main_table, dirs["tables"] / "table1_scenario_dynamics")
    save_table(vignettes, dirs["tables"] / "table_s1_complete_crav_vignettes")
    save_table(qc, dirs["tables"] / "table_s2_data_validation")
    save_table(cells, dirs["tables"] / "table_s3_scenario_level_statistics")
    save_table(transitions, dirs["tables"] / "table_s4_transition_tests")
    save_table(overall, dirs["tables"] / "table_s5_aggregate_level_statistics")
    save_table(anova, dirs["tables"] / "table_s6_repeated_measures_anova")
    save_table(fits, dirs["tables"] / "table_s7_model_fit_comparisons")
    save_table(temps, dirs["tables"] / "table_s8_temperature_robustness")

    figure1_trajectories(
        cells,
        dirs["figures"] / "figure1_crav_trajectories.png",
        dirs["figures"] / "figure1_crav_trajectories.pdf",
    )
    figure_s1_transitions(
        transitions,
        dirs["figures"] / "figure_s1_transition_effects.png",
        dirs["figures"] / "figure_s1_transition_effects.pdf",
    )
    figure_s2_overall(
        overall,
        dirs["figures"] / "figure_s2_aggregate_progression.png",
        dirs["figures"] / "figure_s2_aggregate_progression.pdf",
    )

    summary_text = manuscript_summary_text(df, overall, transitions, scenarios, fits, anova)
    (dirs["checks"] / "manuscript_numbers_check.txt").write_text(summary_text, encoding="utf-8")
    write_metadata(df, files, dirs["checks"] / "analysis_metadata.json")
    write_environment_versions(dirs["checks"] / "environment_versions.txt")

    print(summary_text)
    print(f"\nOutputs written to: {dirs['root'].resolve()}")


if __name__ == "__main__":
    main()
