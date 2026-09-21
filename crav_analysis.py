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
7. Quantifies trajectory-level directional switching/non-monotonicity.
8. Runs a two-way repeated-measures ANOVA (scenario x contextual level).
9. Fits descriptive model specifications comparing R-squared across scenario,
   contextual level, agent identifier, and temperature.
10. Produces Main Figure 1 and Table 1 and Supplementary Figures S1-S2 and
    Tables S1-S10.
11. Writes Supplementary Data S1-S3, validation outputs, analysis metadata,
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


# Theory-guided interpretive map for the 40 stage-to-stage revelations.
#
# IMPORTANT: These labels are descriptive, theory-guided classifications of the
# vignette content. They are not preregistered treatment assignments and are not
# used for inferential testing. In particular, some revelations simultaneously
# introduce considerations that push the reduced-form deservingness index in
# opposing directions. Those transitions are explicitly labelled
# "Mixed/ambiguous" rather than being forced into an increase/decrease prediction.
#
# Keys are (scenario, to_level), so (8, 3) denotes Scenario 8, L2→L3.
TRANSITION_THEORY_MAP: Dict[Tuple[int, int], Dict[str, str]] = {
    (1, 2): {
        "dominant_theoretical_cues": "Greater apparent culpability / norm-relevant conduct",
        "h2_expectation": "Decrease",
        "expectation_basis": "The new information attributes the existing disadvantage to criminal activity, increasing apparent responsibility relative to the sparse baseline.",
        "h3_reclassification_relevance": "High — establishes an initial culpability penalty that later revelations can reinterpret.",
        "interpretation_caveat": "The nature of the criminal activity and degree of personal agency are not specified."
    },
    (1, 3): {
        "dominant_theoretical_cues": "External coercion / reduced agency",
        "h2_expectation": "Increase",
        "expectation_basis": "Forced gang membership reduces apparent voluntariness and introduces an external constraint on the earlier criminal-conduct cue.",
        "h3_reclassification_relevance": "High — directly reinterprets the earlier criminal activity as occurring under coercion.",
        "interpretation_caveat": "The degree and timing of coercion are stated narratively rather than independently manipulated."
    },
    (1, 4): {
        "dominant_theoretical_cues": "Mixed: self-generated gambling debt + continuing constraint",
        "h2_expectation": "Mixed/ambiguous",
        "expectation_basis": "Gambling debt can increase apparent personal responsibility, while 'no other means to pay' preserves a strong constraint. The two components imply opposing directional pressures.",
        "h3_reclassification_relevance": "High — qualifies the prior coercion account by introducing a potentially self-generated antecedent.",
        "interpretation_caveat": "Responsibility and constraint are bundled in the same revelation, so no unique H2 sign is assigned."
    },
    (1, 5): {
        "dominant_theoretical_cues": "Mitigating/prosocial motive + child medical need",
        "h2_expectation": "Increase",
        "expectation_basis": "The gambling is newly linked to financing a child's cancer treatment, adding both a mitigating motive and severe family medical need.",
        "h3_reclassification_relevance": "High — changes how the earlier gambling-debt and criminality sequence can be interpreted.",
        "interpretation_caveat": "Motive and need are introduced together and cannot be separated."
    },

    (2, 2): {
        "dominant_theoretical_cues": "Need / vulnerability / loss",
        "h2_expectation": "Increase",
        "expectation_basis": "Pregnancy loss introduces an adverse outcome and vulnerability without adding a clear responsibility cue.",
        "h3_reclassification_relevance": "Moderate — establishes the initial disadvantage that later levels reinterpret.",
        "interpretation_caveat": "Person A is pregnant at baseline, so the comparison already contains reproductive-status information."
    },
    (2, 3): {
        "dominant_theoretical_cues": "Norm-relevant intentional action with motive not yet specified",
        "h2_expectation": "Mixed/ambiguous",
        "expectation_basis": "The word 'abortion' adds agency and norm-relevant information, but without the reason for the abortion there is no theory-grounded universal direction.",
        "h3_reclassification_relevance": "High — changes the causal account of the earlier pregnancy loss and is itself reinterpreted at later levels.",
        "interpretation_caveat": "Abortion is normatively contested and the vignette does not yet specify medical necessity or motive."
    },
    (2, 4): {
        "dominant_theoretical_cues": "Medical necessity / constraint / life-saving circumstance",
        "h2_expectation": "Increase",
        "expectation_basis": "Life-saving medical necessity sharply reduces the relevance of voluntary-choice interpretations and adds severe need.",
        "h3_reclassification_relevance": "High — directly reclassifies the previously disclosed abortion.",
        "interpretation_caveat": "Medical necessity and life-saving benefit are bundled."
    },
    (2, 5): {
        "dominant_theoretical_cues": "Additional vulnerability / severe contextual complication",
        "h2_expectation": "Increase",
        "expectation_basis": "Incest-related complications add vulnerability and a severe surrounding circumstance to an already medically necessary abortion.",
        "h3_reclassification_relevance": "High — further changes the surrounding moral and causal context.",
        "interpretation_caveat": "The vignette states incest but does not independently specify consent or coercion; those features should not be inferred."
    },

    (3, 2): {
        "dominant_theoretical_cues": "Trauma / vulnerability",
        "h2_expectation": "Increase",
        "expectation_basis": "The communication problem is newly attributed to trauma, adding vulnerability without an explicit culpability cue.",
        "h3_reclassification_relevance": "Moderate — gives a causal account of the baseline condition.",
        "interpretation_caveat": "The source and severity of trauma are not yet specified."
    },
    (3, 3): {
        "dominant_theoretical_cues": "Norm-relevant conduct / possible culpability",
        "h2_expectation": "Decrease",
        "expectation_basis": "Identifying the recipient as an insurgent introduces conduct that may increase apparent responsibility or norm violation relative to unspecified trauma.",
        "h3_reclassification_relevance": "High — changes the social and moral interpretation of the trauma and is later contextualised.",
        "interpretation_caveat": "The label 'insurgent' does not specify role, conduct, voluntariness, or specific wrongdoing."
    },
    (3, 4): {
        "dominant_theoretical_cues": "Bereavement / external trauma / mitigating motive",
        "h2_expectation": "Increase",
        "expectation_basis": "The recipient's joining the insurgency is linked to family deaths in a drone bombing, adding severe bereavement and a mitigating causal context.",
        "h3_reclassification_relevance": "High — directly reinterprets the earlier insurgent cue.",
        "interpretation_caveat": "Bereavement and the stated motive for joining are introduced together."
    },
    (3, 5): {
        "dominant_theoretical_cues": "Mixed: family insurgency involvement + altered interpretation of prior bereavement",
        "h2_expectation": "Mixed/ambiguous",
        "expectation_basis": "Family leadership in the insurgency may weaken the exogeneity or mitigating force of the earlier family-death narrative, but it does not establish the recipient's own culpability.",
        "h3_reclassification_relevance": "High — explicitly changes how the prior family-loss narrative may be understood.",
        "interpretation_caveat": "Responsibility by association should not be equated with the recipient's own conduct."
    },

    (4, 2): {
        "dominant_theoretical_cues": "Greater apparent culpability / controllable norm violation",
        "h2_expectation": "Decrease",
        "expectation_basis": "Drunk driving makes a controllable and norm-relevant contribution to the recipient's own disability salient.",
        "h3_reclassification_relevance": "High — establishes the conduct that later mitigating information can reinterpret.",
        "interpretation_caveat": "The vignette does not quantify intoxication, intent, or broader circumstances at this stage."
    },
    (4, 3): {
        "dominant_theoretical_cues": "Mitigating circumstance / bereavement",
        "h2_expectation": "Increase",
        "expectation_basis": "The drinking is newly linked to a close friend's recent death, adding a mitigating emotional circumstance.",
        "h3_reclassification_relevance": "High — directly recontextualises the earlier drunk-driving disclosure.",
        "interpretation_caveat": "Bereavement mitigates context but does not remove agency for drunk driving."
    },
    (4, 4): {
        "dominant_theoretical_cues": "Additional bereavement / military-service context with indirect relevance to B",
        "h2_expectation": "No directional prediction",
        "expectation_basis": "Learning that the friend died while serving in war adds context to the bereavement, but it does not map cleanly onto B's own need, culpability, constraint, or corrective effort.",
        "h3_reclassification_relevance": "Moderate — elaborates the mitigating context rather than introducing a new action by B.",
        "interpretation_caveat": "The information concerns the friend, so a unique directional prediction for B's allocation would be stronger than the theory supports."
    },
    (4, 5): {
        "dominant_theoretical_cues": "Direct trauma / vulnerability + shared military service",
        "h2_expectation": "Increase",
        "expectation_basis": "The recipient is revealed to have survived the same ambush while serving, adding direct trauma exposure and social contribution.",
        "h3_reclassification_relevance": "High — substantially changes the personal context surrounding the earlier drinking and accident.",
        "interpretation_caveat": "Trauma exposure and military service are introduced jointly."
    },

    (5, 2): {
        "dominant_theoretical_cues": "Prior status / former professional role",
        "h2_expectation": "No directional prediction",
        "expectation_basis": "Former professional-athlete status is socially informative but does not map cleanly onto need, constraint, culpability, mitigation, or corrective effort.",
        "h3_reclassification_relevance": "Low — primarily adds background status information.",
        "interpretation_caveat": "Prior earnings, wealth, effort, and reasons for leaving sport are not specified."
    },
    (5, 3): {
        "dominant_theoretical_cues": "External misfortune / career-ending constraint",
        "h2_expectation": "Increase",
        "expectation_basis": "A career-ending injury provides an external causal explanation for the transition from professional sport to homelessness.",
        "h3_reclassification_relevance": "High — reinterprets the recipient's prior status and current homelessness.",
        "interpretation_caveat": "The cause and preventability of the injury are not specified."
    },
    (5, 4): {
        "dominant_theoretical_cues": "Mixed: health vulnerability/addiction + possible controllability",
        "h2_expectation": "Mixed/ambiguous",
        "expectation_basis": "Painkiller addiction can increase vulnerability while also introducing potentially responsibility-relevant conduct; the vignette does not resolve that tension.",
        "h3_reclassification_relevance": "High — complicates the otherwise exogenous injury narrative.",
        "interpretation_caveat": "The origin, prescription status, and degree of control over the addiction are not specified."
    },
    (5, 5): {
        "dominant_theoretical_cues": "Corrective effort + prosocial/family motive",
        "h2_expectation": "Increase",
        "expectation_basis": "Trying to finance rehabilitation and become a better father adds corrective effort and a family-oriented motive.",
        "h3_reclassification_relevance": "High — reinterprets the addiction cue through attempted rehabilitation.",
        "interpretation_caveat": "Corrective effort and fatherhood motive are bundled."
    },

    (6, 2): {
        "dominant_theoretical_cues": "Greater apparent culpability / norm violation",
        "h2_expectation": "Decrease",
        "expectation_basis": "The recipient's unemployment is newly attributed to theft from an employer, increasing apparent responsibility.",
        "h3_reclassification_relevance": "High — establishes the culpability cue that later effort and need can contextualise.",
        "interpretation_caveat": "The scale, motive, and circumstances of the theft are not yet specified."
    },
    (6, 3): {
        "dominant_theoretical_cues": "Constraint / continuing disadvantage",
        "h2_expectation": "Increase",
        "expectation_basis": "Inability to re-enter the prior field adds an ongoing employment constraint and need.",
        "h3_reclassification_relevance": "Moderate — adds consequences to the earlier theft-based account.",
        "interpretation_caveat": "The constraint may itself be a consequence of earlier culpable conduct, so it is not fully exogenous."
    },
    (6, 4): {
        "dominant_theoretical_cues": "Corrective/work effort",
        "h2_expectation": "Increase",
        "expectation_basis": "Working in fast food despite exclusion from the prior field signals current effort to work.",
        "h3_reclassification_relevance": "High — introduces behaviour inconsistent with a static negative interpretation based solely on the earlier theft.",
        "interpretation_caveat": "The vignette does not measure effort intensity or alternative employment opportunities."
    },
    (6, 5): {
        "dominant_theoretical_cues": "Family need / dependency / responsibility",
        "h2_expectation": "Increase",
        "expectation_basis": "Sole-breadwinner status adds family dependency and need to the recipient's circumstances.",
        "h3_reclassification_relevance": "High — further shifts the narrative from past misconduct toward current responsibility and need.",
        "interpretation_caveat": "Family size and degree of financial need are not specified."
    },

    (7, 2): {
        "dominant_theoretical_cues": "Greater apparent controllability / self-contribution to disadvantage",
        "h2_expectation": "Decrease",
        "expectation_basis": "Years of smoking make a personally controllable contribution to the illness more salient.",
        "h3_reclassification_relevance": "High — provides a clean opportunity to test whether a responsibility cue is invariably penalised.",
        "interpretation_caveat": "Smoking can involve addiction and constrained choice; the vignette does not specify these factors."
    },
    (7, 3): {
        "dominant_theoretical_cues": "External constraint / access to care",
        "h2_expectation": "Increase",
        "expectation_basis": "Lack of insurance adds an external constraint affecting access to treatment.",
        "h3_reclassification_relevance": "Moderate — adds need/constraint after the smoking-responsibility cue.",
        "interpretation_caveat": "The reasons for lacking insurance are not specified."
    },
    (7, 4): {
        "dominant_theoretical_cues": "Financial need / vulnerability",
        "h2_expectation": "Increase",
        "expectation_basis": "Inability to pay medical bills makes financial need more severe and explicit.",
        "h3_reclassification_relevance": "Moderate — deepens the need-based context.",
        "interpretation_caveat": "No information is given about income, assets, or alternative support."
    },
    (7, 5): {
        "dominant_theoretical_cues": "Severe need / treatment-versus-housing trade-off",
        "h2_expectation": "Increase",
        "expectation_basis": "Having to choose between medical treatment and rent intensifies vulnerability and competing basic needs.",
        "h3_reclassification_relevance": "Moderate — deepens the hardship surrounding a partly self-attributed illness.",
        "interpretation_caveat": "The choice is presented narratively; actual costs and feasible alternatives are not specified."
    },

    (8, 2): {
        "dominant_theoretical_cues": "Child medical need / family vulnerability",
        "h2_expectation": "Increase",
        "expectation_basis": "A sick child living in a shelter adds severe dependent need and vulnerability.",
        "h3_reclassification_relevance": "Moderate — adds the family-need context for later conduct.",
        "interpretation_caveat": "The child's illness severity is not quantified."
    },
    (8, 3): {
        "dominant_theoretical_cues": "Mixed: document forgery / norm violation + urgent child medical need / prosocial motive",
        "h2_expectation": "Mixed/ambiguous",
        "expectation_basis": "The same revelation introduces norm-violating conduct and a strong mitigating/prosocial reason for that conduct. H2 therefore supplies no unique sign prediction.",
        "h3_reclassification_relevance": "High — a central case in which the meaning of norm-relevant conduct is inseparable from its motive and constraint.",
        "interpretation_caveat": "This transition should not be coded post hoc as simply 'culpability' or simply 'mitigation'; both are present simultaneously."
    },
    (8, 4): {
        "dominant_theoretical_cues": "Mixed: legal consequence + increased vulnerability",
        "h2_expectation": "Mixed/ambiguous",
        "expectation_basis": "Facing deportation increases vulnerability but is also a consequence of the previously disclosed forgery, leaving the deservingness direction theoretically mixed.",
        "h3_reclassification_relevance": "High — extends the consequences of the earlier mixed-motive conduct.",
        "interpretation_caveat": "The vignette does not specify legal merits, alternatives, or whether deportation would separate the family."
    },
    (8, 5): {
        "dominant_theoretical_cues": "Family loss / vulnerability / conflict-related need",
        "h2_expectation": "Increase",
        "expectation_basis": "A spouse missing in the conflict zone adds family loss, uncertainty, and conflict-related vulnerability.",
        "h3_reclassification_relevance": "Moderate — deepens the hardship context surrounding the earlier conduct.",
        "interpretation_caveat": "The spouse's circumstances and their material implications are not specified."
    },

    (9, 2): {
        "dominant_theoretical_cues": "External shock / bad luck",
        "h2_expectation": "Increase",
        "expectation_basis": "Loss of the main investor due to recession attributes the startup's difficulty to an external macroeconomic shock.",
        "h3_reclassification_relevance": "Moderate — provides an exogenous causal account of the baseline risk of failure.",
        "interpretation_caveat": "The startup's underlying viability before the recession is not specified."
    },
    (9, 3): {
        "dominant_theoretical_cues": "Mixed: persistence/effort + self-chosen financial risk",
        "h2_expectation": "Mixed/ambiguous",
        "expectation_basis": "Taking a high-interest loan can signal effort to preserve the business while simultaneously increasing personally chosen financial risk.",
        "h3_reclassification_relevance": "High — changes the balance between external shock and self-directed response.",
        "interpretation_caveat": "Loan terms, available alternatives, and decision quality are not specified."
    },
    (9, 4): {
        "dominant_theoretical_cues": "Mixed: financial vulnerability + consequence of prior risk-taking",
        "h2_expectation": "Mixed/ambiguous",
        "expectation_basis": "Default and personal bankruptcy increase need and vulnerability but also follow the previously disclosed high-interest borrowing decision.",
        "h3_reclassification_relevance": "High — makes the consequences of the mixed external/choice pathway salient.",
        "interpretation_caveat": "The relative contribution of recession, loan choice, and business decisions cannot be separated."
    },
    (9, 5): {
        "dominant_theoretical_cues": "Corrective/prosocial effort / social contribution",
        "h2_expectation": "Increase",
        "expectation_basis": "Trying to pivot toward a youth social enterprise introduces a prosocial goal and corrective effort.",
        "h3_reclassification_relevance": "High — reframes the failing business through a prospective social-contribution motive.",
        "interpretation_caveat": "The feasibility and sincerity of the proposed pivot are not independently verified."
    },

    (10, 2): {
        "dominant_theoretical_cues": "Mitigating/prosocial motive + family need",
        "h2_expectation": "Increase",
        "expectation_basis": "The offence is newly linked to unemployment and feeding the family, adding need and a mitigating family-oriented motive.",
        "h3_reclassification_relevance": "High — directly reinterprets the baseline ex-convict label through the motive for the offence.",
        "interpretation_caveat": "The type and severity of the offence are not specified."
    },
    (10, 3): {
        "dominant_theoretical_cues": "Family loss / housing vulnerability",
        "h2_expectation": "Increase",
        "expectation_basis": "Family separation and transitional housing add vulnerability and social loss.",
        "h3_reclassification_relevance": "Moderate — deepens the post-incarceration hardship context.",
        "interpretation_caveat": "The reasons for family separation are not specified."
    },
    (10, 4): {
        "dominant_theoretical_cues": "External pressure / constraint to re-offend",
        "h2_expectation": "Increase",
        "expectation_basis": "Pressure from former criminal associates introduces an external constraint and threat to rehabilitation without stating that B has re-offended.",
        "h3_reclassification_relevance": "High — changes the interpretation of future criminal risk from purely voluntary to socially constrained.",
        "interpretation_caveat": "The degree of coercion versus ordinary social pressure is not specified."
    },
    (10, 5): {
        "dominant_theoretical_cues": "Corrective effort + prosocial contribution",
        "h2_expectation": "Increase",
        "expectation_basis": "Joining rehabilitation and volunteering provide direct evidence of corrective effort and social contribution.",
        "h3_reclassification_relevance": "High — reclassifies the recipient's current trajectory relative to the baseline ex-convict label.",
        "interpretation_caveat": "The duration and effectiveness of rehabilitation/volunteering are not specified."
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
                "n_agents": n,
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
    """Aggregate over scenarios within matched agent identifier before inference."""
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
            "n_agents": n,
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
    """Compute 40 within-agent level-to-level changes and paired tests.

    Wilcoxon signed-rank tests are the primary inferential tests. Paired t-tests
    are reported as a parametric robustness check only where the paired
    differences have non-zero variance. For zero-variance non-zero transitions,
    the conventional paired t statistic is not estimable and is stored as NaN.
    """
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
                t_stat = np.nan
                t_p = np.nan
                t_estimable = False
            else:
                t_result = stats.ttest_rel(after, before, nan_policy="omit")
                t_stat, t_p = float(t_result.statistic), float(t_result.pvalue)
                t_estimable = True

            try:
                w_result = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
                w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
            except ValueError:
                # This occurs only if all paired differences are exactly zero.
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
                    "paired_t_estimable": t_estimable,
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
                        "scenario_name": SCENARIO_NAMES[scenario],
                        "from_level": from_level,
                        "to_level": to_level,
                        "transition": f"L{from_level}→L{to_level}",
                        "new_information": cue,
                        "delta_B_pp": float(value),
                    }
                )

    results = pd.DataFrame(rows)

    # BH correction for the paired t-tests is applied only to estimable tests.
    results["paired_t_q_BH"] = np.nan
    mask = results["paired_t_p"].notna()
    if mask.any():
        results.loc[mask, "paired_t_q_BH"] = multipletests(
            results.loc[mask, "paired_t_p"], method="fdr_bh"
        )[1]

    # Wilcoxon tests are estimable for all observed non-zero transition patterns.
    results["wilcoxon_q_BH"] = multipletests(results["wilcoxon_p"], method="fdr_bh")[1]
    results["significant_FDR_05"] = results["wilcoxon_q_BH"] < 0.05
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
        levels = c["level"].to_numpy(dtype=int)
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

        min_i = int(np.argmin(level_means))
        max_i = int(np.argmax(level_means))
        rows.append(
            {
                "scenario": scenario,
                "scenario_name": SCENARIO_NAMES[scenario],
                "L1_mean_B_pct": float(level_means[0]),
                "L5_mean_B_pct": float(level_means[-1]),
                "net_change_pp": float(level_means[-1] - level_means[0]),
                "lowest_level": int(levels[min_i]),
                "lowest_mean_B_pct": float(level_means[min_i]),
                "highest_level": int(levels[max_i]),
                "highest_mean_B_pct": float(level_means[max_i]),
                "mean_abs_reallocation_pp": float(np.mean(np.abs(diffs))),
                "negative_transitions": int((t["mean_delta_pp"] < 0).sum()),
                "trajectory": trajectory,
                "largest_fall": largest_fall,
                "largest_fall_pp": largest_fall_pp,
                "largest_rise": largest_rise,
                "largest_rise_pp": largest_rise_pp,
            }
        )
    return pd.DataFrame(rows)

def trajectory_switching(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise within-trajectory directional switching.

    A scenario-specific conversational trajectory is classified as mixed-direction
    when its four stage-to-stage changes contain at least one increase and one
    decrease. Directional reversals count sign changes among consecutive non-zero
    deltas; zero changes are ignored when determining the sign sequence.
    """
    wide = df.pivot(index=["agent", "scenario"], columns="level", values="allocation_B_pct")
    rows = []
    for (agent, scenario), r in wide.iterrows():
        deltas = np.array([r[l] - r[l - 1] for l in range(2, EXPECTED_LEVELS + 1)], dtype=float)
        signs = [1 if x > 0 else -1 for x in deltas if not np.isclose(x, 0.0)]
        n_reversals = sum(a != b for a, b in zip(signs, signs[1:]))
        has_pos = bool(np.any(deltas > 0))
        has_neg = bool(np.any(deltas < 0))
        if has_pos and has_neg:
            direction_class = "mixed-direction"
        elif has_pos:
            direction_class = "non-decreasing"
        elif has_neg:
            direction_class = "non-increasing"
        else:
            direction_class = "no change"
        total_abs = float(np.abs(deltas).sum())
        net = float(r[EXPECTED_LEVELS] - r[1])
        reversal_component = float(total_abs - abs(net))
        rows.append({
            "agent": agent,
            "scenario": int(scenario),
            "scenario_name": SCENARIO_NAMES[int(scenario)],
            "direction_class": direction_class,
            "mixed_direction": has_pos and has_neg,
            "directional_reversals": int(n_reversals),
            "total_absolute_reallocation_pp": total_abs,
            "net_change_pp": net,
            "reversal_component_pp": reversal_component,
            "reversal_share": reversal_component / total_abs if total_abs > 0 else 0.0,
        })
    trajectory = pd.DataFrame(rows)

    summary_rows = []
    for scenario, g in trajectory.groupby("scenario", sort=True):
        summary_rows.append({
            "scenario": int(scenario),
            "scenario_name": SCENARIO_NAMES[int(scenario)],
            "n_trajectories": int(len(g)),
            "mixed_direction_n": int(g["mixed_direction"].sum()),
            "mixed_direction_pct": float(100 * g["mixed_direction"].mean()),
            "mean_directional_reversals": float(g["directional_reversals"].mean()),
            "median_directional_reversals": float(g["directional_reversals"].median()),
            "mean_total_absolute_reallocation_pp": float(g["total_absolute_reallocation_pp"].mean()),
            "mean_net_change_pp": float(g["net_change_pp"].mean()),
            "mean_reversal_component_pp": float(g["reversal_component_pp"].mean()),
            "mean_reversal_share_pct": float(100 * g["reversal_share"].mean()),
        })
    return trajectory, pd.DataFrame(summary_rows)


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
        ("Agent identifier only", "allocation_B_pct ~ C(agent)"),
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
            "n_agents": int(n_agents),
            "n_allocation_observations": int((df["temperature"] == temp).sum()),
            "mean_agent_B_pct": mean,
            "sd_agent_B_pct": sd,
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

def hypothesis_evidence_table(
    transitions: pd.DataFrame,
    switching: pd.DataFrame,
) -> pd.DataFrame:
    """Build the high-level hypothesis-to-evidence map (Supplementary Table S9).

    H1 has direct transition-level inference. H2 and H3 are theory-guided,
    descriptive pattern-consistency hypotheses because the vignette dimensions
    and reveal order are not independently randomised.

    Numerical values are read from the current analysis outputs rather than
    hard-coded, so Table S9 remains synchronised with the analysed data.
    """
    n_sig = int((transitions["wilcoxon_q_BH"] < 0.05).sum())
    min_delta = float(transitions["mean_delta_pp"].min())
    max_delta = float(transitions["mean_delta_pp"].max())

    def delta(scenario: int, to_level: int) -> float:
        row = transitions.loc[
            (transitions["scenario"] == scenario)
            & (transitions["to_level"] == to_level),
            "mean_delta_pp",
        ]
        if len(row) != 1:
            raise ValueError(
                f"Expected exactly one transition for scenario={scenario}, "
                f"to_level={to_level}; found {len(row)}."
            )
        return float(row.iloc[0])

    mixed_n = int(switching["mixed_direction_n"].sum())
    traj_n = int(switching["n_trajectories"].sum())
    mixed_pct = 100.0 * mixed_n / traj_n if traj_n else np.nan

    return pd.DataFrame([
        {
            "hypothesis": "H1: Contextual updating",
            "observable_implication": (
                "Morally diagnostic additions produce non-zero average "
                "stage-to-stage reallocations within the repeated "
                "conversational trajectories."
            ),
            "evidence": (
                f"{n_sig}/40 stage-to-stage transitions were distinguishable "
                "from zero after Benjamini–Hochberg correction in the primary "
                f"Wilcoxon tests; mean changes ranged from {min_delta:.1f} to "
                f"+{max_delta:.1f} percentage points. This is repeated-output "
                "evidence within one model configuration, not population "
                "inference about humans or independently trained models."
            ),
        },
        {
            "hypothesis": "H2: Deservingness-sensitive directionality",
            "observable_implication": (
                "When a revelation is dominated by greater apparent culpability "
                "or controllable conduct, B's share is expected to fall; when it "
                "is dominated by need, external constraint, mitigating "
                "circumstances, or corrective effort, B's share is expected to "
                "rise. Revelations that bundle opposing considerations have a "
                "mixed/ambiguous theoretical prediction."
            ),
            "evidence": (
                "Several clear transitions match the proposed direction: "
                f"criminal activity ({delta(1, 2):+.1f} pp), drunk driving "
                f"({delta(4, 2):+.1f} pp), and workplace theft "
                f"({delta(6, 2):+.1f} pp) generated large penalties, whereas "
                f"forced gang membership ({delta(1, 3):+.2f} pp), life-saving "
                f"medical necessity ({delta(2, 4):+.1f} pp), bereavement "
                f"({delta(4, 3):+.1f} pp), sole-breadwinner status "
                f"({delta(6, 5):+.1f} pp), and rehabilitation/volunteering "
                f"({delta(10, 5):+.1f} pp) increased support. The pattern is "
                f"not invariant: the smoking revelation moved support by "
                f"{delta(7, 2):+.1f} pp, and several transitions (for example, "
                "document forgery for urgent child medical care) combine "
                "opposing cues and are therefore not assigned a unique H2 sign. "
                "Table S10 provides the full transition-level map."
            ),
        },
        {
            "hypothesis": "H3: Narrative reclassification",
            "observable_implication": (
                "Later information can change how earlier facts are interpreted, "
                "producing directional reversals and different consequences of "
                "superficially similar culpability- or norm-relevant cues across "
                "narratives."
            ),
            "evidence": (
                f"Across all {traj_n} scenario-specific trajectories, "
                f"{mixed_n} ({mixed_pct:.1f}%) contained both positive and "
                "negative reallocations. Scenarios 1, 4, and 6 show clear "
                "penalty-and-restoration paths, while superficially similar "
                "responsibility-relevant cues have different consequences across "
                "narratives (for example, criminal activity, drunk driving, and "
                "workplace theft reduce support, whereas the smoking revelation "
                "increases it). Fixed reveal order means narrative content "
                "cannot be separated from anchoring or path dependence."
            ),
        },
    ])


def transition_hypothesis_map(transitions: pd.DataFrame) -> pd.DataFrame:
    """Build the descriptive theory-guided transition evidence map (Table S10).

    The H2 cue/expectation labels are interpretive classifications of vignette
    content. They are not preregistered and are not used for inferential tests.
    "Mixed/ambiguous" and "No directional prediction" are retained whenever the
    new revelation does not support a unique directional expectation.

    Direction consistency is reported only for clean Increase/Decrease entries.
    It is intentionally not converted into an accuracy score or hypothesis test.
    """
    expected_keys = {
        (scenario, to_level)
        for scenario in range(1, EXPECTED_SCENARIOS + 1)
        for to_level in range(2, EXPECTED_LEVELS + 1)
    }
    actual_keys = set(TRANSITION_THEORY_MAP)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "TRANSITION_THEORY_MAP must contain exactly the 40 stage-to-stage "
            f"transitions. Missing={missing}; extra={extra}"
        )

    valid_expectations = {
        "Increase",
        "Decrease",
        "Mixed/ambiguous",
        "No directional prediction",
    }
    invalid = {
        key: value["h2_expectation"]
        for key, value in TRANSITION_THEORY_MAP.items()
        if value["h2_expectation"] not in valid_expectations
    }
    if invalid:
        raise ValueError(f"Invalid H2 expectation labels in theory map: {invalid}")

    rows = []
    for _, r in transitions.sort_values(["scenario", "to_level"]).iterrows():
        key = (int(r["scenario"]), int(r["to_level"]))
        theory = TRANSITION_THEORY_MAP[key]
        expectation = theory["h2_expectation"]
        observed = str(r["direction"])

        if expectation == "Increase":
            consistent = "Yes" if observed == "increase" else "No"
        elif expectation == "Decrease":
            consistent = "Yes" if observed == "decrease" else "No"
        else:
            consistent = "Not classified"

        rows.append({
            "scenario": int(r["scenario"]),
            "scenario_name": r["scenario_name"],
            "transition": r["transition"],
            "new_information": r["new_information"],
            "dominant_theoretical_cues": theory["dominant_theoretical_cues"],
            "H2_directional_expectation": expectation,
            "expectation_basis": theory["expectation_basis"],
            "observed_delta_B_pp": round(float(r["mean_delta_pp"]), 2),
            "observed_direction": observed,
            "direction_consistent_with_H2": consistent,
            "H3_reclassification_relevance": theory["h3_reclassification_relevance"],
            "interpretation_caveat": theory["interpretation_caveat"],
        })
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, stem: Path, index: bool = False) -> None:
    """Save CSV plus a readable Markdown version."""
    df.to_csv(stem.with_suffix(".csv"), index=index)
    try:
        stem.with_suffix(".md").write_text(df.to_markdown(index=index), encoding="utf-8")
    except Exception:
        # to_markdown requires tabulate; CSV remains canonical if unavailable.
        pass


def figure1_trajectories(df: pd.DataFrame, cells: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    """Main figure: individual trajectories plus scenario means and 95% CIs."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.8), sharex=True, sharey=True)
    axes = axes.ravel()

    y_min = max(0, math.floor((df["allocation_B_pct"].min() - 5) / 5) * 5)
    y_max = min(100, math.ceil((df["allocation_B_pct"].max() + 5) / 5) * 5)

    for scenario, ax in zip(range(1, 11), axes):
        raw_s = df[df["scenario"] == scenario]
        for _, g in raw_s.groupby("agent", sort=False):
            g = g.sort_values("level")
            ax.plot(g["level"], g["allocation_B_pct"], linewidth=0.55, alpha=0.12)

        c = cells[cells["scenario"] == scenario].sort_values("level")
        x = c["level"].to_numpy()
        y = c["mean_B_pct"].to_numpy()
        yerr = np.vstack([y - c["ci95_low"].to_numpy(), c["ci95_high"].to_numpy() - y])

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.8, zorder=5)
        ax.axhline(50, linestyle="--", linewidth=0.9)
        ax.set_title(f"S{scenario}. {PLOT_TITLES[scenario]}", fontsize=9)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(["L1", "L2", "L3", "L4", "L5"])
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.18)

        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)

    for ax in axes[5:]:
        ax.set_xlabel("Sequential contextual level")
    for ax in axes[::5]:
        ax.set_ylabel("Allocation to Person B (%)")

    fig.suptitle("CRAV allocation trajectories across sequential contextual revelations", fontsize=13, y=0.995)
    fig.text(
        0.5, 0.012,
        "Thin lines show the 50 scenario-specific stochastic trajectories; points and error bars show means and 95% t confidence intervals. "
        "Dashed line denotes an equal 50/50 split.",
        ha="center", fontsize=8.5,
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


def figure_s2_switching(switch_summary: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    """Supplement: proportion of mixed-direction scenario-specific trajectories."""
    t = switch_summary.sort_values("scenario")
    x = np.arange(len(t))
    y = t["mixed_direction_pct"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(s)}" for s in t["scenario"]])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Mixed-direction trajectories (%)")
    ax.set_xlabel("CRAV scenario")
    ax.set_title("Within-trajectory directional switching")
    ax.grid(axis="y", alpha=0.18)
    for b, val in zip(bars, y):
        ax.text(b.get_x() + b.get_width()/2, val + 2.0, f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)

def manuscript_summary_text(
    df: pd.DataFrame,
    overall: pd.DataFrame,
    transitions: pd.DataFrame,
    scenarios: pd.DataFrame,
    switching: pd.DataFrame,
    model_fits: pd.DataFrame,
    anova: pd.DataFrame,
) -> str:
    agent_level = df.groupby(["agent", "level"])["allocation_B_pct"].mean().unstack()
    l1_l5_delta = agent_level[5] - agent_level[1]
    d_mean, d_sd, d_lo, d_hi, d_n = mean_ci_t(l1_l5_delta)
    d_test = stats.ttest_rel(agent_level[5], agent_level[1])

    n_negative = int((transitions["mean_delta_pp"] < 0).sum())
    n_positive = int((transitions["mean_delta_pp"] > 0).sum())
    n_sig_w = int((transitions["wilcoxon_q_BH"] < 0.05).sum())
    n_t_estimable = int(transitions["paired_t_estimable"].sum())
    n_sig_t = int((transitions.loc[transitions["paired_t_estimable"], "paired_t_q_BH"] < 0.05).sum())
    n_nonmono_mean = int((scenarios["trajectory"] == "non-monotonic").sum())
    n_mixed = int(switching["mixed_direction_n"].sum())
    n_traj = int(switching["n_trajectories"].sum())

    fit_lookup = model_fits.set_index("model")
    lines = [
        "CRAV MANUSCRIPT NUMERICAL CROSS-CHECK",
        "=" * 55,
        f"Observations: {len(df):,}; agent identifiers: {df['agent'].nunique()}; scenario-specific trajectories: {n_traj}; scenarios: 10; levels: 5.",
        f"Model(s): {', '.join(map(str, df['model'].dropna().unique()))}.",
        f"Collection window: {df['created'].min()} to {df['created'].max()} (UTC).",
        "",
        "Overall contextual progression",
        f"Level 1 mean B allocation: {overall.loc[overall.level == 1, 'mean_B_pct'].iloc[0]:.3f}%.",
        f"Level 5 mean B allocation: {overall.loc[overall.level == 5, 'mean_B_pct'].iloc[0]:.3f}%.",
        f"Matched agent-ID mean L5-L1 change: {d_mean:.3f} pp (95% CI {d_lo:.3f}, {d_hi:.3f}); paired t({d_n-1})={d_test.statistic:.3f}, p={d_test.pvalue:.3e}.",
        "",
        "Transition structure",
        f"{n_sig_w}/40 Wilcoxon transition tests significant at BH-FDR q<.05; {n_positive} positive and {n_negative} negative mean changes.",
        f"Paired t tests estimable for {n_t_estimable}/40 transitions; {n_sig_t}/{n_t_estimable} estimable tests significant after BH-FDR correction.",
        f"Non-monotonic scenario-mean trajectories: {n_nonmono_mean}/10.",
        f"Mixed-direction individual trajectories: {n_mixed}/{n_traj} ({100*n_mixed/n_traj:.1f}%).",
        "",
        "Descriptive R-squared",
    ]
    for name in [
        "Temperature only", "Agent identifier only", "Scenario only", "Contextual level only",
        "Scenario + level", "Scenario × level", "Scenario × level + temperature",
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
    trajectory_detail, switching = trajectory_switching(df)
    anova = repeated_measures_anova(df)
    fits = model_fit_table(df)
    temps = temperature_summary(df)
    vignettes = vignette_table(df)
    transition_map = transition_hypothesis_map(transitions)

    delta_long.to_csv(dirs["data"] / "supplementary_data_s2_transition_deltas.csv", index=False)
    trajectory_detail.to_csv(dirs["data"] / "supplementary_data_s3_trajectory_switching.csv", index=False)

    main_table = scenarios[[
        "scenario", "scenario_name", "lowest_level", "lowest_mean_B_pct",
        "highest_level", "highest_mean_B_pct", "mean_abs_reallocation_pp",
        "largest_fall", "largest_rise",
    ]].copy()
    main_table["lowest_mean_B"] = main_table.apply(lambda r: f"L{int(r.lowest_level)}; {r.lowest_mean_B_pct:.1f}%", axis=1)
    main_table["highest_mean_B"] = main_table.apply(lambda r: f"L{int(r.highest_level)}; {r.highest_mean_B_pct:.1f}%", axis=1)
    main_table = main_table[[
        "scenario", "scenario_name", "lowest_mean_B", "highest_mean_B",
        "mean_abs_reallocation_pp", "largest_fall", "largest_rise",
    ]]

    save_table(main_table, dirs["tables"] / "table1_scenario_dynamics")
    save_table(vignettes, dirs["tables"] / "table_s1_complete_crav_vignettes")
    save_table(qc, dirs["tables"] / "table_s2_data_validation")
    save_table(cells, dirs["tables"] / "table_s3_scenario_level_statistics")
    save_table(transitions, dirs["tables"] / "table_s4_transition_tests")
    save_table(anova, dirs["tables"] / "table_s5_repeated_measures_anova")
    save_table(fits, dirs["tables"] / "table_s6_model_fit_comparisons")
    save_table(switching, dirs["tables"] / "table_s7_trajectory_switching")
    save_table(temps, dirs["tables"] / "table_s8_temperature_robustness")
    save_table(hypothesis_evidence_table(transitions, switching), dirs["tables"] / "table_s9_hypothesis_evidence")
    save_table(transition_map, dirs["tables"] / "table_s10_transition_hypothesis_map")

    figure1_trajectories(
        df, cells,
        dirs["figures"] / "figure1_crav_trajectories.png",
        dirs["figures"] / "figure1_crav_trajectories.pdf",
    )
    figure_s1_transitions(
        transitions,
        dirs["figures"] / "figure_s1_transition_effects.png",
        dirs["figures"] / "figure_s1_transition_effects.pdf",
    )
    figure_s2_switching(
        switching,
        dirs["figures"] / "figure_s2_directional_switching.png",
        dirs["figures"] / "figure_s2_directional_switching.pdf",
    )

    summary_text = manuscript_summary_text(df, overall, transitions, scenarios, switching, fits, anova)
    (dirs["checks"] / "manuscript_numbers_check.txt").write_text(summary_text, encoding="utf-8")
    write_metadata(df, files, dirs["checks"] / "analysis_metadata.json")
    write_environment_versions(dirs["checks"] / "environment_versions.txt")

    print(summary_text)
    print(f"\nOutputs written to: {dirs['root'].resolve()}")


if __name__ == "__main__":
    main()
