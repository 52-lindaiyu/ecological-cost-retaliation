# The ecological cost of retaliation bounds decentralised reciprocity in commons with tipping points

This repository contains the simulation and analysis code for the paper of the same title. The code implements a threshold-regenerating common-pool resource, theoretical calculations, best-response certification of management rules, independent tabular multi-agent Q-learning, and evolutionary tournaments.

The study uses simulated data; no external dataset is required. The scripts write generated CSV and JSON files to `results/`.

## Requirements

- Python 3.11 or newer
- Packages listed in `requirements.txt` (`numpy` and `pandas`)

From the repository root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Code structure

| Path | Purpose |
| --- | --- |
| `commons/config.py` | Baseline and alternative model parameters |
| `commons/core.py` | Resource dynamics, payoffs, and theoretical quantities |
| `commons/exploit.py` | Candidate rules and same-model best-response certification |
| `commons/marl.py`, `commons/marl_batch.py` | Policy evaluation and batched independent Q-learning |
| `commons/evo.py` | Evolutionary population dynamics |
| `run_*.py` | Experiment drivers |
| `make_distilled.py` | Distilled learned rules used in certification and tournaments |
| `tests_model.py` | Baseline and resource-conservation checks |

## Run the code

Run commands from the repository root. First create the output directory and run the model checks:

```bash
python -c "from pathlib import Path; Path('results').mkdir(exist_ok=True)"
python tests_model.py
```

Run the main computations in this order:

```bash
python run_theory.py
python run_planner_cert.py
python run_exploit.py
python run_marl.py
python make_distilled.py
python run_contestants.py
python run_review.py --with-c
python run_review2.py
python run_robust.py
python run_evo.py
```

`run_theory.py` calculates phase and sanction-window quantities. `run_exploit.py` certifies interpretable rules and sensitivity comparisons. `run_marl.py` trains and evaluates learned policies; `make_distilled.py` then creates the contestant rules needed by `run_contestants.py` and `run_evo.py`. `run_review.py` and `run_review2.py` perform learning replications and additional controls. `run_robust.py` evaluates the factorial robustness grid. Outputs are written under `results/`.

The baseline parameters are defined in `commons/config.py`. The objective sweep in `run_marl.py` evaluates each learned policy on 300 rollouts; its other sweeps use 200. Large sweeps and certification runs may take substantial time.

## Paper

This code accompanies the paper *The ecological cost of retaliation bounds decentralised reciprocity in commons with tipping points*. The final article citation and link can be added here when publication details are available.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
