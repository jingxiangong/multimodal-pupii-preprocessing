# Multimodal Pupillometry Preprocessing

Python functions for preprocessing pupil-diameter signals and integrating them with facial-expression measurements from iMotions 10 exports.

This repository accompanies the research article **“A Preprocessing Pipeline for Pupillometry Signal from Multimodal iMotion Data”**, published in *Sensors* (2025).

**Paper:** https://doi.org/10.3390/s25154737

## Overview

Eye-tracking recordings can contain blink-related gaps, abrupt measurement artifacts, invalid pupil values, and repeated timestamps. In the study, pupil-diameter measurements (approximately 120 Hz) also needed to be compared with facial-expression measurements (approximately 30 Hz).

The Python workflow prepares the pupil signal for subsequent research analysis. It is a preprocessing and data-integration method, not a clinical diagnostic tool or a cognitive-load prediction model.

## Processing workflow

```text
iMotions 10 CSV export
  -> Select pupil measurements and timestamps
  -> Check valid pupil values and remove duplicate timestamps
  -> Detect artifacts using MAD and a moving-average criterion
  -> Interpolate eligible gaps with PCHIP
  -> Estimate selected missing measurements using the other eye
  -> Calculate and normalize mean pupil diameter
  -> Aggregate pupil data and join facial-expression measurements
  -> Export a processed CSV and plot diagnostic signals
```

### Main methods

- **Artifact filtering:** A Median Absolute Deviation (MAD) threshold on pupil-dilation speed is followed by a moving-average deviation check.
- **Interpolation:** PCHIP is used for its shape-preserving behavior and observed error trade-off. The code avoids filling gaps longer than 250 ms.
- **Binocular estimation:** Linear regression estimates some missing measurements in one eye when corresponding observations from both eyes are available for fitting.
- **Normalization:** Pupil dilation is normalized relative to a selected 400 ms pre-event reference window.
- **Multimodal integration:** The processed pupil signal is aggregated toward 30 Hz and merged with selected facial-expression measurements.

The 1000 Hz interpolation grid is a computational time grid, not a claim that the sensor collected 1000 observations per second.

## Repository files

| File | Purpose |
| --- | --- |
| [`Pupil_Preprocessing_Functions.py`](Pupil_Preprocessing_Functions.py) | Preprocessing functions, interpolation comparison, data integration, and plots. |
| [`Example.py`](Example.py) | Example of processing one iMotions CSV export. |
| [`requirements.txt`](requirements.txt) | Python dependencies used by the example. |

## Setup

The documented research environment used Python 3.11.11 and iMotions 10 (version 10.1.38911.4). Install the dependencies in a virtual environment.

On Windows PowerShell with standard Windows Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the example

1. Obtain an **authorized iMotions 10 CSV export**. No participant-level data or `data.csv` is included in this repository.
2. Open `Example.py` and set `INPUT_CSV`, `START_TIME_MS`, `END_TIME_MS`, and `EMOTION_COLUMNS` for your recording.
3. Run the script from the repository root:

```powershell
.\.venv\Scripts\python.exe Example.py
```

The script generates plots, prints a preview of the processed data, and writes `result.csv` to the working directory.

**Input format:** The loader expects the iMotions export layout used in the study, including the original preamble and columns corresponding to timestamp, left pupil diameter, and right pupil diameter. A generic three-column CSV will not work without adapting the loader.

**Parameters:** The example includes recording-specific timestamps and uses different MAD multipliers for the two eyes (`n=20` for the left, `n=16` for the right). Do not treat them as universal defaults for other datasets.

## Research results and scope

The published workflow was applied to pupil recordings from **25 participants**. The paper uses representative signal traces and a masked-point interpolation comparison to explain the steps and the method selection. For PCHIP, Akima, and Makima, the comparison considered both RMSE and maximum absolute error; PCHIP was selected for its overall error and shape-preservation trade-off.

This public repository provides the preprocessing functions and a **single-recording example**, not the full historical multi-participant processing environment. The example has not been independently demonstrated to regenerate every figure or numerical result in the published paper.

The method also has limitations: a pre-event reference window is not a controlled baseline; reconstructing a missing signal requires sufficient observations; and timestamp aggregation and cross-modal alignment should be checked for each dataset. The workflow does not validate clinical effectiveness.

## Publication

Ong, J., He, W., Maglanque, P., Jiang, X., Gillman, L. M., Vergis, A., & Hardy, K. (2025). **A Preprocessing Pipeline for Pupillometry Signal from Multimodal iMotion Data.** *Sensors*, **25**(15), 4737. https://doi.org/10.3390/s25154737

## Data and reuse

The repository does not contain participant data. Use only recordings you are authorized to access. The paper's open-access license does not automatically provide a software license for the Python code in this repository.
