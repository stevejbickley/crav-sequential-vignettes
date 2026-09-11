# CRAV Sequential Vignettes

Reproducibility materials for:

**Bickley, S. J., Chan, H. F., Stadelmann, D., Tani, M., & Torgler, B. (2026). _Context-Sensitive Resource Allocation: Evidence from Sequential Vignettes Using Artificial Agents_.**

## Overview

This study introduces **Contextual Resource Allocation Vignettes (CRAV)**, a sequential vignette framework for examining how artificial agents revise resource-allocation decisions as additional contextual information is progressively revealed.

Across **10 five-stage vignettes**, the allocation problem remains fixed at **$1,000** while information about a single recipient is revealed sequentially. We use **50 stochastic GPT-4o conversational trajectories per vignette**, yielding **500 scenario-specific trajectories and 2,500 allocation decisions**.

CRAV captures stage-to-stage moral and distributive updating that is typically hidden in one-shot economic games. Results are specific to the model and inference configuration used and should **not be interpreted as a human benchmark**.

## Data Generation

The artificial-agent responses analysed in this study were generated using [SurveyLM](https://surveylm.panalogy-lab.com/Platform) and exported as 10 scenario-level CSV files.

The archived exports in `input_data/` are the starting point for the reproducibility workflow. SurveyLM is not required to reproduce the analyses reported in the paper, and this repository does not attempt to guarantee exact regeneration of the original stochastic model responses.

## Repository Structure

```text
.
├── crav_analysis.py
├── requirements.txt
├── README.md
├── CITATION.cff
├── LICENSE
├── .gitignore
│
├── input_data/
│   ├── crav_vignette_01_raw.csv
│   ├── crav_vignette_02_raw.csv
│   ├── crav_vignette_03_raw.csv
│   ├── crav_vignette_04_raw.csv
│   ├── crav_vignette_05_raw.csv
│   ├── crav_vignette_06_raw.csv
│   ├── crav_vignette_07_raw.csv
│   ├── crav_vignette_08_raw.csv
│   ├── crav_vignette_09_raw.csv
│   └── crav_vignette_10_raw.csv
│
├── data/
│   ├── supplementary_data_s1_crav_long_clean.csv
│   └── supplementary_data_s2_transition_deltas.csv
│
├── figures/
│   ├── figure1_crav_trajectories.pdf
│   ├── figure1_crav_trajectories.png
│   ├── figure_s1_transition_effects.pdf
│   ├── figure_s1_transition_effects.png
│   ├── figure_s2_aggregate_progression.pdf
│   └── figure_s2_aggregate_progression.png
│
├── tables/
│   ├── table1_scenario_dynamics.csv
│   ├── table1_scenario_dynamics.md
│   ├── table_s1_complete_crav_vignettes.csv
│   ├── table_s1_complete_crav_vignettes.md
│   ├── table_s2_data_validation.csv
│   ├── table_s2_data_validation.md
│   ├── table_s3_scenario_level_statistics.csv
│   ├── table_s3_scenario_level_statistics.md
│   ├── table_s4_transition_tests.csv
│   ├── table_s4_transition_tests.md
│   ├── table_s5_aggregate_level_statistics.csv
│   ├── table_s5_aggregate_level_statistics.md
│   ├── table_s6_repeated_measures_anova.csv
│   ├── table_s6_repeated_measures_anova.md
│   ├── table_s7_model_fit_comparisons.csv
│   ├── table_s7_model_fit_comparisons.md
│   ├── table_s8_temperature_robustness.csv
│   └── table_s8_temperature_robustness.md
│
└── checks/
    ├── analysis_metadata.json
    ├── environment_versions.txt
    └── manuscript_numbers_check.txt
```   
    
### Main files

- **`crav_analysis.py`** — main analysis and reproducibility script, including data parsing and validation, descriptive and inferential analyses, and generation of the reported tables and figures.
- **`input_data/`** — the 10 archived scenario-level SurveyLM CRAV exports used as inputs to the analysis.
- **`data/`** — Supplementary Data S1–S2: the cleaned long-format CRAV panel and model-instance-level stage-to-stage allocation changes.
- **`figures/`** — Main Figure 1 and Supplementary Figures S1–S2 in PDF and PNG formats.
- **`tables/`** — Main Table 1 and Supplementary Tables S1–S8 in CSV and Markdown formats, including data-validation and robustness outputs.
- **`checks/`** — automatically generated reproducibility and verification outputs, including analysis metadata, recorded software versions, and a numerical cross-check of key manuscript results.

## Reproducing the Analysis

From the repository root, create and activate a Python virtual environment.

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
```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then run:

```bash
python crav_analysis.py
```

When finished, deactivate the virtual environment with:
```bash
deactivate
```

By default, the script reads the 10 crav_vignette_*_raw.csv files in input_data/ and reproduces the cleaned datasets, statistical analyses, tables, figures, and analysis checks contained in the repository.

The `checks/` directory contains automatically generated reproducibility and verification information, including analysis metadata, recorded software versions, and a numerical cross-check of key manuscript results.

Because the artificial agent responses are stochastic and hosted model implementations may change over time, the repository is designed to reproduce the reported analysis from the archived model outputs rather than to guarantee exact regeneration of the original GPT-4o responses.

## Citation

If you use this repository or its materials, please cite the associated study:

> Bickley, S. J., Chan, H. F., Stadelmann, D., Tani, M., & Torgler, B. (2026). *Context-Sensitive Resource Allocation: Evidence from Sequential Vignettes Using Artificial Agents.*

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff), with the associated manuscript specified as the preferred citation. GitHub uses this file to provide repository citation information.

The citation metadata will be updated with the journal reference and DOI following publication.

## Licence

The code in this repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

## Authors

**Steve J. Bickley** — Queensland University of Technology; ARC Industrial Transformation Training Centre for Behavioural Insights for Technology Adoption; Panalogy Lab Pty Ltd  
**Ho Fai Chan** — Queensland University of Technology; ARC Industrial Transformation Training Centre for Behavioural Insights for Technology Adoption; Panalogy Lab Pty Ltd  
**David Stadelmann** — University of Bayreuth; CREMA – Centre for Research in Economics, Management, and the Arts  
**Massimiliano Tani** — University of New South Wales  
**Benno Torgler** — Queensland University of Technology; ARC Industrial Transformation Training Centre for Behavioural Insights for Technology Adoption; Panalogy Lab Pty Ltd; CREMA – Centre for Research in Economics, Management, and the Arts

**Correspondence:** Steve J. Bickley — s.bickley@qut.edu.au
