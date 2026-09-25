# CRAV Sequential Vignettes

Reproducibility materials for:

**Bickley, S. J., Chan, H. F., Stadelmann, D., Tani, M., & Torgler, B. (2026). _Context-Sensitive Resource Allocation: Evidence from Sequential Vignettes Using Artificial Agents_.**

## Overview

This study introduces **Contextual Resource Allocation Vignettes (CRAV)**, a sequential-vignette framework for examining how resource-allocation decisions change as morally and socially relevant contextual information is progressively revealed.

Across **10 five-stage vignettes**, the allocation problem remains fixed at **USD $1,000** while information about Person B is revealed sequentially. The archived study contains **50 stochastic GPT-4o conversational trajectories per vignette**, yielding **500 scenario-specific trajectories and 2,500 allocation decisions**. A fresh conversational thread was initiated for each scenario; conversational state was retained across Levels 1–5 within a scenario but not between scenarios.

The analysis includes a trajectory-level measure of **directional switching**. A scenario-specific trajectory is classified as *mixed-direction* when its four successive changes contain at least one increase and at least one decrease in the allocation to Person B. This complements the scenario-mean trajectories by showing whether non-monotonic updating is present within stochastic conversational runs rather than being produced only by aggregation.

An optional supplementary analysis uses **outcome-blinded multi-model contextual coding**. Three heterogeneous AI coders apply the same frozen 1–7 coding protocol to the 50 cumulative vignette states without receiving GPT-4o allocations, transition effects, scenario/level labels, figures, or the existing theory map. The resulting ratings operationalise responsibility/controllability, need/vulnerability, external constraint/coercion, mitigating circumstances/motive, and corrective/prosocial effort for theory-guided analysis of the 40 contextual transitions.

## Data generation

Responses were generated using [SurveyLM](https://surveylm.panalogy-lab.com/Platform) and exported as 10 scenario-level CSV files. SurveyLM is not required to reproduce the analyses reported in the paper. The archived CSV exports in `input_data/` are the starting point for both the main reproducibility workflow and the optional blinded contextual-coding workflow.

Because hosted model implementations are stochastic and can change over time, this repository reproduces the reported analyses from the archived outputs; it does not guarantee exact regeneration of the original GPT-4o responses or future re-generation of the same blinded coder ratings.

## Repository structure

```text
.
├── crav_analysis.py
├── crav_blinded_context_coding.py
├── crav_context_coding_codebook.json
├── crav_context_coding_codebook.md
├── requirements.txt
├── README.md
├── CITATION.cff
├── LICENSE
├── .gitignore
├── .env.example
│
├── input_data/
│   ├── crav_vignette_01_raw.csv
│   ├── ...
│   └── crav_vignette_10_raw.csv
│
├── data/
│   ├── supplementary_data_s1_crav_long_clean.csv
│   ├── supplementary_data_s2_transition_deltas.csv
│   ├── supplementary_data_s3_trajectory_switching.csv
│   ├── supplementary_data_s4_blinded_context_codings.csv      # if coding is run
│   ├── supplementary_data_s5_context_state_scores.csv         # if coding is run
│   └── supplementary_data_s6_coder_transition_scores.csv      # if coding is run
│
├── figures/
│   ├── figure1_crav_trajectories.pdf/.png
│   ├── figure2_context_coding_association.pdf/.png             # if coding is run
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
│   ├── table_s7_temperature_robustness.csv/.md
│   ├── table_s8_trajectory_switching.csv/.md
│   ├── table_s9_hypothesis_evidence.csv/.md
│   ├── table_s10_transition_hypothesis_map.csv/.md
│   ├── table_s11_blinded_coder_reliability.csv/.md             # if coding is run
│   ├── table_s12_context_coding_transition_map.csv/.md          # if coding is run
│   └── table_s13_context_coding_associations.csv/.md            # if coding is run
│
└── checks/
    ├── analysis_metadata.json
    ├── environment_versions.txt
    ├── manuscript_numbers_check.txt
    ├── blinded_context_coding_manifest.csv                      # if coding is prepared/run
    ├── blinded_context_coding_order.csv                         # if coding is prepared/run
    └── blinded_context_coding_metadata.json                     # if coding is prepared/run
```

If the conceptual CRAV-extensions table is retained in the Supporting Information, it follows the empirical coding tables as **Table S14** and is not generated by `crav_analysis.py`.

## Main analyses

`crav_analysis.py` performs the complete reproducible workflow:

1. loads the 10 archived SurveyLM exports;
2. parses Person A/B allocations from the free-text answers;
3. validates the fixed USD $1,000 budget and balanced 10 × 5 × 50 design;
4. creates the cleaned long-format panel;
5. calculates scenario-level allocation trajectories;
6. calculates all 40 stage-to-stage within-trajectory reallocations;
7. applies two-sided Wilcoxon signed-rank tests with Benjamini-Hochberg false-discovery-rate correction;
8. reports paired t-tests as a parametric robustness check when the paired-difference variance is non-zero;
9. quantifies trajectory-level mixed-direction switching and sign reversals;
10. runs the overall repeated-measures analysis and descriptive model-fit/temperature diagnostics;
11. constructs the theory-guided H1–H3 evidence map and complete 40-transition H2/H3 map;
12. if blinded coding data are present, evaluates coder reliability and the association between theory-coded contextual change and observed mean reallocation;
13. reproduces all analysis-generated empirical tables and figures; and
14. writes reproducibility checks and software-version metadata.

### Directional-switching definitions

For each of the 500 scenario-specific trajectories, the script calculates the four stage-to-stage changes in Person B's budget share.

- **Mixed-direction trajectory:** at least one positive and at least one negative stage-to-stage change.
- **Directional reversals:** sign changes between consecutive non-zero stage-to-stage changes; zero changes are ignored for sign-sequence counting.
- **Total absolute reallocation:** sum of the absolute values of the four stage-to-stage changes.
- **Net change:** Level 5 allocation minus Level 1 allocation.

The primary switching result is that **247 of 500 trajectories (49.4%) are mixed-direction**. Scenario-specific rates are reported in **Table S8 and Figure S2**.

## Blinded contextual coding

### Frozen codebook

`crav_context_coding_codebook.json` is the canonical machine-readable coding protocol. `crav_context_coding_codebook.md` is its human-readable companion. The coding script records the codebook version and SHA-256 hash in every coding output so that all coder results can be tied to the same frozen definitions.

The five coded dimensions are:

- responsibility / controllability;
- need / vulnerability;
- external constraint / coercion;
- mitigating circumstances / motive; and
- corrective / prosocial effort.

Each dimension is rated on the same pre-specified 1–7 rubric across providers. The derived externality/luck score is `8 - responsibility_controllability`, and the equal-weight contextual index is the mean of the oriented five components. The index is a theory-guided descriptive measure, not an estimated latent construct or structural utility parameter.

### Provider consistency

The same canonical codebook is used across all providers, but each provider's native structured-output interface is used:

- **OpenAI:** the full frozen codebook is supplied with a strict Pydantic structured-output schema.
- **Anthropic:** the same full frozen codebook is supplied and a forced tool call must satisfy the same schema.
- **TypeSafe AI / Jev:** each of the five dimensions is submitted as an ordered seven-level `Score` question using the exact same level definitions and shared coding rules. Jev returns typed score distributions rather than generated prose, so the modal level is mapped back to the common integer 1–7 rubric.

The provider interfaces are therefore not mechanically identical, but the **unit of coding, dimension definitions, scale anchors, blinding rules, and final 1–7 variables are held constant**. Inter-coder agreement is evaluated empirically rather than assumed.

### How the 50 blinded states are constructed

The 50 cumulative vignette states do **not** need to be entered manually. `crav_blinded_context_coding.py` reads the archived files in `input_data/`, extracts the `question id` and cumulative `question` text, and reduces the 2,500 response rows to the **50 unique scenario × level vignette states**.

Scenario and level identifiers are retained **locally only** so that returned ratings can later be merged back to the CRAV data. They are never included in the API request. For each coder, the script creates a deterministic coder-specific random order from the recorded seed. The coding API receives only:

1. the frozen coding rubric; and
2. the cumulative narrative text for the current vignette state.

It does **not** receive the scenario number, level number, GPT-4o allocation, transition effect, Figure 1, or existing theory-map classification.

For auditability, the script writes:

- `checks/blinded_context_coding_manifest.csv`: the local state-to-study mapping;
- `checks/blinded_context_coding_order.csv`: the random presentation order used for each coder; and
- `checks/blinded_context_coding_metadata.json`: codebook hash, coder specifications, seed, input source, and blinding description.

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

### API keys for blinded coding

Create an `.env` file and add your own provider keys:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
TYPESAFE_API_KEY=...
```

`.env` is excluded by `.gitignore` and should never be committed. Existing shell/CI environment variables take precedence over `.env` values.

The default coder panel uses one model from each provider:

```text
openai:gpt-5.6-sol
anthropic:claude-opus-5
typesafe:jev-latest
```

To inspect the 50-state manifest and randomised coder orders **without making any API calls**, run:

```bash
python crav_blinded_context_coding.py --prepare-only
```

To run the default three-provider blinded coding:

```bash
python crav_blinded_context_coding.py
```

An alternative OpenAI/Anthropic-only panel can be supplied explicitly, for example:

```bash
python crav_blinded_context_coding.py \
  --coders openai:gpt-5.6-sol anthropic:claude-opus-5 anthropic:claude-sonnet-5
```

The coding script writes results incrementally and resumes incomplete runs by default. Use `--overwrite` only when an intentionally fresh coding run is required.

### Run the full analysis

After the blinded coding file exists, run:

```bash
python crav_analysis.py
```

The default configuration reads `input_data/crav_vignette_*_raw.csv`. If `data/supplementary_data_s4_blinded_context_codings.csv` exists, the contextual-coding analysis and Figure 2 / Tables S11–S13 are included automatically. If the file is absent, the original CRAV analysis still runs normally.

## Blinded-coding analysis

The coding analysis uses the **40 unique within-scenario transitions** as its primary empirical unit rather than treating the 2,000 agent-by-transition observations as independent contextual treatments.

The pre-specified coding analysis reports:

- ordinal Krippendorff's alpha for each coded dimension;
- an equal-weight theory-guided contextual index;
- the change in that index between successive cumulative states;
- Spearman association between theory-coded contextual change and observed mean reallocation;
- 95% scenario-cluster bootstrap intervals obtained by resampling the ten scenarios;
- directional consensus when at least two-thirds of coders agree on the sign of the index change; and
- sensitivity analyses using responsibility/externality alone, individual positive dimensions, and leave-one-dimension-out indices.

The original GPT-4o outcomes were known to the authors when this coding protocol was developed. The coding analysis should therefore be described as **outcome-blinded and theory-guided, not preregistered confirmation or independent human validation**.

## Reproducibility outputs

- **Supplementary Data S1:** cleaned long-format allocation panel.
- **Supplementary Data S2:** agent-by-transition stage-to-stage reallocations.
- **Supplementary Data S3:** trajectory-level directional-switching diagnostics.
- **Supplementary Data S4:** raw outcome-blinded multi-model contextual ratings, if coding is run.
- **Supplementary Data S5:** aggregated cumulative-state contextual scores, if coding is run.
- **Supplementary Data S6:** coder-level transition score changes, if coding is run.
- **Figure S1 / Table S4:** complete stage-to-stage transition effects and inference.
- **Figure S2 / Table S8:** trajectory-level directional-switching results.
- **Tables S6–S7:** descriptive model-fit and sampling-temperature diagnostics.
- **Table S9:** high-level hypothesis-to-evidence map linking H1–H3 to their observable implications and corresponding CRAV evidence.
- **Table S10:** complete author theory-guided transition evidence map. Mixed/ambiguous and No directional prediction transitions are retained rather than being forced into increase/decrease predictions.
- **Figure 2 / Tables S11–S13:** blinded contextual-coding reliability, transition mapping, and association/sensitivity results, if coding is run.
- **`checks/manuscript_numbers_check.txt`:** compact cross-check of key manuscript values.
- **`checks/environment_versions.txt`:** software versions used for the packaged run.
- **`checks/blinded_context_coding_metadata.json`:** coding protocol hash, coder specifications, randomisation seed, and blinding metadata.

## Interpretation

The 500 CRAV trajectories are stochastic conversational runs from one underlying GPT-4o configuration, not human participants or 500 independently trained models. The same set of 50 agent identifiers was reused across scenarios for the balanced design, but each scenario began a fresh conversational thread. Inferential statistics therefore address reproducibility across repeated model outputs under the archived prompting and sampling configuration; they do not estimate the prevalence of corresponding preferences in a human population.

Tables S9 and S10 provide theory-guided mappings between the CRAV results and H1–H3. H1 is evaluated using the transition-level inferential tests. H2 and H3 are pattern-consistency hypotheses because contextual dimensions and reveal order are bundled rather than independently randomised. The cue classifications in Table S10 are interpretive mappings of vignette content, not independently estimated latent constructs.

The blinded multi-model coding analysis, when included, provides a separate outcome-blinded operationalisation of theory-relevant contextual dimensions. The AI coders are heterogeneous model coders rather than independent human raters; shared training data or model-family regularities may induce dependence. Inter-coder reliability and sensitivity analyses are therefore reported explicitly.

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
