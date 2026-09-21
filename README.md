# CRAV Sequential Vignettes

Reproducibility materials for:

**Bickley, S. J., Chan, H. F., Stadelmann, D., Tani, M., & Torgler, B. (2026). _Context-Sensitive Resource Allocation: Evidence from Sequential Vignettes Using Artificial Agents_.**

## Overview

This study introduces **Contextual Resource Allocation Vignettes (CRAV)**, a sequential-vignette framework for examining how resource-allocation decisions change as morally and socially relevant contextual information is progressively revealed.

Across **10 five-stage vignettes**, the allocation problem remains fixed at **USD $1,000** while information about Person B is revealed sequentially. The archived study contains **50 stochastic GPT-4o conversational trajectories per vignette**, yielding **500 scenario-specific trajectories and 2,500 allocation decisions**. A fresh conversational thread was initiated for each scenario; conversational state was retained across Levels 1–5 within a scenario but not between scenarios.

The analysis also includes a trajectory-level measure of **directional switching**. A scenario-specific trajectory is classified as *mixed-direction* when its four successive changes contain at least one increase and at least one decrease in the allocation to Person B. This analysis complements the scenario-mean trajectories by showing whether non-monotonic updating is present within stochastic conversational runs rather than being produced only by aggregation.

## Data generation

Responses were generated using [SurveyLM](https://surveylm.panalogy-lab.com/Platform) and exported as 10 scenario-level CSV files. SurveyLM is not required to reproduce the analyses reported in the paper. The archived CSV exports in `input_data/` are the starting point for the reproducibility workflow.

Because hosted model implementations are stochastic and can change over time, this repository reproduces the reported analyses from the archived outputs; it does not guarantee exact regeneration of the original GPT-4o responses.

## Repository structure

```text
.
├── crav_analysis.py
├── requirements.txt
├── README.md
├── CITATION.cff
├── LICENSE
│
├── input_data/
│   ├── crav_vignette_01_raw.csv
│   ├── ...
│   └── crav_vignette_10_raw.csv
│
├── data/
│   ├── supplementary_data_s1_crav_long_clean.csv
│   ├── supplementary_data_s2_transition_deltas.csv
│   └── supplementary_data_s3_trajectory_switching.csv
│
├── figures/
│   ├── figure1_crav_trajectories.pdf/.png
│   ├── figure_s1_transition_effects.pdf/.png
│   └── figure_s2_directional_switching.pdf/.png
│
├── tables/
│   ├── table1_scenario_dynamics.csv/.md
│   ├── table_s1_complete_crav_vignettes.csv/.md
│   ├── table_s2_data_validation.csv/.md
│   ├── table_s3_scenario_level_statistics.csv/.md
│   ├── table_s4_transition_tests.csv/.md
│   ├── table_s5_repeated_measures_anova.csv/.md
│   ├── table_s6_model_fit_comparisons.csv/.md
│   ├── table_s7_trajectory_switching.csv/.md
│   ├── table_s8_temperature_robustness.csv/.md
│   ├── table_s9_hypothesis_evidence.csv/.md
│   └── table_s10_transition_hypothesis_map.csv/.md
│
└── checks/
    ├── analysis_metadata.json
    ├── environment_versions.txt
    └── manuscript_numbers_check.txt
```

## Main analyses

`crav_analysis.py` performs the complete reproducible workflow:

1. loads the 10 archived SurveyLM exports;
2. parses Person A/B allocations from the free-text answers;
3. validates the fixed USD $1,000 budget and balanced 10 × 5 × 50 design;
4. creates the cleaned long-format panel;
5. calculates scenario-level allocation trajectories;
6. calculates all 40 stage-to-stage within-trajectory reallocations;
7. applies two-sided Wilcoxon signed-rank tests with Benjamini-Hochberg false-discovery-rate correction;
8. reports paired t-tests as a parametric robustness check when the paired-difference variance is non-zero (five zero-variance transitions are correctly reported as not estimable);
9. quantifies trajectory-level mixed-direction switching and sign reversals; 
10. runs the overall repeated-measures analysis and descriptive model-fit/temperature diagnostics; 
11. constructs the theory-guided H1–H3 evidence map and the complete 40-transition H2/H3 map, explicitly retaining Mixed/ambiguous and No directional prediction classifications where no unique directional prediction is warranted; 
12. reproduces all paper and Online Appendix tables and figures; and 
13. writes reproducibility checks and software-version metadata.

### Directional-switching definitions

For each of the 500 scenario-specific trajectories, the script calculates the four stage-to-stage changes in Person B's budget share.

- **Mixed-direction trajectory:** at least one positive and at least one negative stage-to-stage change.
- **Directional reversals:** sign changes between consecutive non-zero stage-to-stage changes; zero changes are ignored for sign-sequence counting.
- **Total absolute reallocation:** sum of the absolute values of the four stage-to-stage changes.
- **Net change:** Level 5 allocation minus Level 1 allocation.

The paper's primary new switching result is that **247 of 500 trajectories (49.4%) are mixed-direction**. Scenario-specific rates are reported in Table S7 and Figure S2.

## Reproducing the analysis

Create and activate a Python virtual environment from the repository root, then install the dependencies.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (Command Prompt)

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run:

```bash
python crav_analysis.py
```

The default configuration reads `input_data/crav_vignette_*_raw.csv` and writes the cleaned data, analyses, tables, figures, and checks into the repository folders shown above.

## Reproducibility outputs

- **Supplementary Data S1:** cleaned long-format allocation panel.
- **Supplementary Data S2:** agent-by-transition stage-to-stage reallocations.
- **Supplementary Data S3:** trajectory-level directional-switching diagnostics.
- **Table S4:** full transition results, including the newly introduced information at each transition and correct handling of zero-variance paired t-tests.
- **Figure S2 / Table S7:** trajectory-level directional-switching results.
- **Table S9:** high-level hypothesis-to-evidence map linking H1–H3 to their observable implications and corresponding CRAV evidence.
- **Table S10:** complete transition-level theory-guided evidence map linking each of the 40 contextual revelations to its dominant theoretical cue(s), H2 directional expectation, observed reallocation, directional consistency where applicable, H3 relevance, and interpretation caveat. Mixed/ambiguous and No directional prediction transitions are retained rather than being forced into increase/decrease predictions. The classifications are descriptive and theory-guided, not preregistered treatment assignments or additional inferential tests.
- **`checks/manuscript_numbers_check.txt`:** compact cross-check of the key values reported in the manuscript.
- **`checks/environment_versions.txt`:** software versions used for the packaged run.

## Interpretation

The 500 trajectories are stochastic conversational runs from one underlying GPT-4o configuration, not human participants or 500 independently trained models. The same set of 50 agent identifiers was reused across scenarios for the balanced design, but each scenario began a fresh conversational thread. Inferential statistics therefore address reproducibility across repeated model outputs under the archived prompting and sampling configuration; they do not estimate the prevalence of corresponding preferences in a human population.

Tables S9 and S10 provide theory-guided mappings between the CRAV results and H1–H3. H1 is evaluated using the transition-level inferential tests. H2 and H3 are pattern-consistency hypotheses because contextual dimensions and reveal order are bundled rather than independently randomised. The cue classifications in Table S10 are therefore interpretive mappings of vignette content, not independently estimated latent constructs. Directional consistency is evaluated only where the theoretical mapping supports a clear increase or decrease expectation; mixed or theoretically indeterminate transitions are left unclassified.

## Citation

If you use this repository or its materials, please cite the associated study:

> Bickley, S. J., Chan, H. F., Stadelmann, D., Tani, M., & Torgler, B. (2026). *Context-Sensitive Resource Allocation: Evidence from Sequential Vignettes Using Artificial Agents.*

Machine-readable metadata are provided in `CITATION.cff`. They should be updated with the final journal citation and DOI following publication.

## Licence

Analysis code is released under the MIT License. The raw model-output data and manuscript materials remain subject to any applicable third-party platform or publisher terms.

## Authors

- **Steve J. Bickley** — Queensland University of Technology; ARC Industrial Transformation Training Centre for Behavioural Insights for Technology Adoption; Panalogy Lab Pty Ltd
- **Ho Fai Chan** — Queensland University of Technology; ARC Industrial Transformation Training Centre for Behavioural Insights for Technology Adoption; Panalogy Lab Pty Ltd
- **David Stadelmann** — University of Bayreuth; CREMA – Centre for Research in Economics, Management, and the Arts
- **Massimiliano Tani** — University of New South Wales
- **Benno Torgler** — Queensland University of Technology; ARC Industrial Transformation Training Centre for Behavioural Insights for Technology Adoption; Panalogy Lab Pty Ltd; CREMA – Centre for Research in Economics, Management, and the Arts

**Correspondence:** Steve J. Bickley — s.bickley@qut.edu.au
