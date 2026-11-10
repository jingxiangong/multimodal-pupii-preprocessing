# Multimodal Pupillometry Preprocessing

**Python workflow for cleaning pupil-diameter signals and preparing them for multimodal biomedical research.**

This repository accompanies the peer-reviewed study *A Preprocessing Pipeline for Pupillometry Signal from Multimodal iMotion Data* (Ong et al., *Sensors*, 2025). It provides research-oriented Python functions and an example script for preprocessing eye-tracking measurements exported from **iMotions 10** and combining them with selected facial-expression measurements.

The study used recordings from simulated endoscopy. The goal of this code is to prepare more interpretable, temporally comparable research data—not to diagnose medical conditions or predict cognitive load.

**[Read the paper](https://doi.org/10.3390/s25154737)** · **[View the published figures and methods](https://www.mdpi.com/1424-8220/25/15/4737)**

## Research context

Pupil-diameter recordings can contain invalid measurements, blink-related artifacts, abrupt tracking errors, duplicate timestamps, and gaps that differ between the left and right eyes. Combining pupil recordings with facial-expression measurements adds a timing challenge because the two modalities have different sampling rates (approximately **120 Hz** for pupil diameter and **30 Hz** for facial expressions in the study).

The published work documents a preprocessing sequence that addresses these issues before subsequent analysis. The workflow was applied to recordings from **25 participants**; the paper uses one participant's recording for its detailed step-by-step illustrations. This is a methodological demonstration, not a clinical validation study or a benchmark of workload prediction.

## Processing workflow

```text
iMotions 10 CSV export
  -> Select timestamps and bilateral pupil-diameter columns
  -> Remove invalid values and duplicate timestamps
  -> Filter dilation-speed outliers using MAD
  -> Filter remaining local deviations using a moving-average criterion
  -> Interpolate eligible gaps using PCHIP (computational 1000 Hz grid)
  -> Estimate selected single-eye missing values using linear regression
  -> Calculate mean pupil diameter and baseline-normalized dilation
  -> Aggregate toward ~30 Hz and outer-join facial-expression records
  -> Export research-ready CSV and generate inspection plots
```

### Key methodological choices

| Decision | Reason |
| --- | --- |
| **MAD followed by moving-average filtering** | Targets both large dilation-speed anomalies and deviations from the local signal trend. |
| **PCHIP interpolation** | Selected for a trade-off between reconstruction error and shape preservation, not solely the lowest RMSE. The interpolation functions omit reconstructed points corresponding to source gaps longer than **250 ms**. |
| **Binocular linear regression** | Estimates some values missing in only one eye when paired observations are available; it cannot recover times when neither eye is measured. |
| **Baseline normalization and multimodal integration** | Uses a selected valid **400 ms** pre-event window, then aggregates pupil values toward **30 Hz** for comparison with facial-expression measurements. |

The **1000 Hz** interpolation grid is a numerical representation of the signal, **not** a new 1000 Hz physical recording. An outer join preserves unmatched timestamps; it does not by itself prove exact synchronization.

## Repository contents

| File | Purpose |
| --- | --- |
| [`Pupil_Preprocessing_Functions.py`](Pupil_Preprocessing_Functions.py) | Cleaning, MAD/MA filtering, interpolation, binocular estimation, normalization, integration, plotting, and an interpolation-evaluation helper. |
| [`Example.py`](Example.py) | Example sequence showing how to apply the functions to an iMotions CSV export. |
| [`requirements.txt`](requirements.txt) | Pinned library versions used by the public example. |

This is a small research-code repository, **not** a packaged command-line application or a dataset release.

## Requirements

The documented research environment used **Python 3.11.11** and **iMotions 10 (10.1.38911.4)**. The pinned Python packages are:

- pandas 2.2.3
- NumPy 2.2.4
- SciPy 1.15.2
- scikit-learn 1.6.1
- matplotlib 3.10.1

Install dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## How to use the example

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/jingxiangong/multimodal-pupii-preprocessing.git
   cd multimodal-pupii-preprocessing
   ```

2. Create and activate a Python virtual environment if desired, then install the requirements:

   ```bash
   python -m venv .venv
   # macOS / Linux:
   source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Obtain an **authorized iMotions 10 CSV export** with the relevant pupil and facial-expression channels. This repository **does not include `data.csv` or participant-level source data**.

4. Open [`Example.py`](Example.py) and adjust its configuration to your dataset before running it:

   - `pd.read_csv('data.csv')`: point to your own permitted input file.
   - `start_time` and `end_time`: specify the annotated task interval in the dataset's timestamp units.
   - `selected_columns`: match the facial-expression channels in your particular iMotions export.
   - Optional plot limits: the `plot_left_pupil(...)` calls contain illustrative axis ranges that may not fit your recording.

5. Run the example:

   ```bash
   python Example.py
   ```

The script displays diagnostic plots, prints a preview of the merged DataFrame, and writes **`result.csv`** to the working directory. It does not automatically regenerate every published figure or table.

### Expected input format

The current loader is **specific to the iMotions CSV layout used in the study**. In particular, the implementation accesses source columns such as `Unnamed: 1`, `Unnamed: 84`, and `Unnamed: 85`, removes an export preamble, and expects the resulting columns `Timestamp`, `ET_PupilLeft`, and `ET_PupilRight`. Other export templates may need an adapted loader; a generic three-column CSV is **not** interchangeable with the raw export assumed by `Example.py`.

Some functions also require sufficient valid, ordered measurements and overlapping left/right observations. Missing or differently formatted inputs may cause the example to fail rather than produce an automatic fallback.

## Validation and evidence boundaries

The paper illustrates filtering and interpolation with signal plots and compares interpolation methods through a **randomly masked-point reconstruction experiment**. That experiment is informative for method selection, but it does not establish performance on every naturally occurring blink, long tracking-loss gap, or unseen eye-tracker export.

The publicly available code should also be read with these limits in mind:

- **Not the complete historical batch runner.** The repository provides a single-input example; its exact correspondence to the private multi-participant processing scripts used during the study has not been independently established.
- **Early missing-row filtering.** The present `process_pupil_data()` implementation drops some incomplete rows before later one-eye reconstruction, so the regression step cannot be assumed to recover every original unilateral missing observation.
- **Baseline is a proxy.** The study did not record a dedicated controlled baseline; the workflow selects a valid 400 ms pre-event window.
- **Timestamp alignment needs checking.** The aggregation and outer-join implementation do not guarantee that every pupil and facial-expression bin has identical timestamp keys.
- **Example parameters are recording-specific.** In the published example script, the MAD multiplier is `n=20` for the left eye and `n=16` for the right eye. Those are the current script values, not a universal parameter recommendation or a claim that the public example exactly reproduces every reported result.
- **No clinical performance claim.** This repository is not a diagnostic device or a validated prediction system.

For the formally reported methods and findings, consult the published paper. Treat the present repository as a research code example, not a turnkey reproduction package.

## Publication and citation

Ong, J., He, W., Maglanque, P., Jiang, X., Gillman, L. M., Vergis, A., & Hardy, K. (2025). **A Preprocessing Pipeline for Pupillometry Signal from Multimodal iMotion Data**. *Sensors*, **25**(15), 4737. https://doi.org/10.3390/s25154737

The paper was published on **31 July 2025**. Its published author-contributions statement documents collaborative methodological/software work and identifies Jingxiang Ong's contributions to formal analysis, data curation, visualization, and original manuscript drafting.

## Data and reuse

The repository does not distribute raw participant data. Use only data you are authorized to access and share. The journal article and its figures have their own published open-access terms; those terms **do not automatically define a software license for this repository's Python code**. No standalone code license is currently specified in the repository.
