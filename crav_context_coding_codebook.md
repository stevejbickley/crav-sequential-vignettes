# CRAV Blinded Contextual Coding Protocol

**Version:** 1.0.0  
**Frozen:** 25 September 2026  
**Canonical machine-readable file:** `crav_context_coding_codebook.json`

## Unit of coding

One cumulative CRAV vignette state. Coders see Person A and Person B exactly as described at that state, but receive no scenario number, level number, allocation outcome, transition effect, figure, or existing theory classification. Coders rate **Person B only** using stated information.

## General rules

- Do not predict, recommend, or infer a resource allocation.
- Do not infer facts that are not stated in the vignette.
- Do not assume that an action is voluntary, coerced, blameworthy, admirable, or avoidable unless the text supports that interpretation.
- Rate each dimension independently. A vignette may score high on more than one dimension, including dimensions with opposing theoretical implications.
- Use integer scores from 1 to 7 only.
- For responsibility/controllability, use 4 when the causal role of Person B's controllable choices is genuinely unclear or not specified. For the other four dimensions, use 1 when there is no stated evidence of that feature.

## Dimensions

### Responsibility / controllability

To what extent are Person B's adverse circumstances or focal disadvantage attributable to Person B's voluntary or controllable choices?

| Score | Definition |
|---:|---|
| 1 | Almost entirely external, accidental, imposed, or outside Person B’s control. |
| 2 | Mostly external or outside Person B’s control, with little stated voluntary contribution. |
| 3 | More external than controllable, although some personal contribution is stated or plausible from the text. |
| 4 | Mixed, genuinely unclear, or insufficient information about controllability. |
| 5 | More attributable to Person B’s controllable choices than to external circumstances. |
| 6 | Mostly attributable to Person B’s voluntary or controllable choices. |
| 7 | Strongly attributable to Person B’s voluntary or controllable choices. |

**Theory orientation:** Higher values predict lower allocation to Person B, all else equal.

### Need / vulnerability

How severe is the need, hardship, loss, or vulnerability explicitly described for Person B?

| Score | Definition |
|---:|---|
| 1 | No salient need, hardship, loss, or vulnerability is stated. |
| 2 | Slight need or vulnerability. |
| 3 | Mild-to-moderate need or vulnerability. |
| 4 | Moderate need or vulnerability. |
| 5 | Substantial need, hardship, loss, or vulnerability. |
| 6 | Severe need, hardship, loss, or vulnerability. |
| 7 | Extreme or acute need, hardship, loss, or vulnerability. |

**Theory orientation:** Higher values predict higher allocation to Person B, all else equal.

### External constraint / coercion

To what extent do external forces, coercion, restricted alternatives, or circumstances outside Person B's control constrain Person B's situation or choices?

| Score | Definition |
|---:|---|
| 1 | No meaningful external constraint, coercion, or restricted alternatives are stated. |
| 2 | Slight external constraint or some restriction of alternatives. |
| 3 | Mild-to-moderate external constraint. |
| 4 | Moderate or mixed external constraint. |
| 5 | Substantial external constraint or restricted alternatives. |
| 6 | Strong external constraint or coercive pressure. |
| 7 | Strong coercion, a clearly forced hand, or severely restricted alternatives. |

**Theory orientation:** Higher values predict higher allocation to Person B, all else equal.

### Mitigating circumstances / motive

To what extent does the vignette provide an explicit circumstance or motive that mitigates how negatively Person B's conduct or situation might otherwise be interpreted?

| Score | Definition |
|---:|---|
| 1 | No explicit mitigating circumstance, necessity, or prosocial motive is stated. |
| 2 | Slight mitigating context. |
| 3 | Some mitigation, but limited or qualified. |
| 4 | Moderate or mixed mitigation. |
| 5 | Substantial mitigating circumstance, necessity, or prosocial motive. |
| 6 | Strong mitigation or necessity. |
| 7 | Very strong mitigating circumstance, necessity, or prosocial motive. |

**Theory orientation:** Higher values predict higher allocation to Person B, all else equal.

### Corrective / prosocial effort

To what extent does Person B explicitly demonstrate corrective effort, rehabilitation, constructive work, restitution, or contribution to others?

| Score | Definition |
|---:|---|
| 1 | No corrective, rehabilitative, constructive, restitutive, or prosocial effort is stated. |
| 2 | Slight evidence of corrective or prosocial effort. |
| 3 | Some corrective or prosocial effort, but limited or early. |
| 4 | Moderate or partial corrective/prosocial effort. |
| 5 | Substantial corrective, constructive, or prosocial effort. |
| 6 | Strong corrective, rehabilitative, restitutive, or prosocial effort. |
| 7 | Very strong and explicit corrective effort, rehabilitation, restitution, or prosocial contribution. |

**Theory orientation:** Higher values predict higher allocation to Person B, all else equal.

## Derived scores

`externality_luck_score = 8 - responsibility_controllability`

`contextual_deservingness_index = mean(externality_luck_score, need_vulnerability, external_constraint_coercion, mitigating_circumstances_motive, corrective_prosocial_effort)`

The composite is an equal-weight theory-guided descriptive index. It is **not** an estimated latent construct or structural utility parameter.

## Pre-specified analysis

- Reliability: ordinal Krippendorff’s alpha for each coded dimension.
- Primary empirical unit: the **40 unique within-scenario transitions**.
- Primary association: Spearman correlation between change in the aggregated contextual index and observed mean reallocation to Person B.
- Uncertainty: scenario-cluster bootstrap resampling the ten scenarios.
- Directional coding: derive each coder’s index change mechanically. A positive or negative directional expectation is assigned when at least two-thirds of coders agree on the sign. Otherwise it is mixed/indeterminate.
- Sensitivity: responsibility/externality alone, each positive component separately, and leave-one-component-out indices.

The coding analysis is supplementary and theory-guided. Because the original allocation outcomes were already known to the authors when this coding protocol was developed, it should not be represented as preregistered confirmation. The outcome-blind coding procedure reduces post-hoc classification risk but does not make the analysis equivalent to independent human validation.
