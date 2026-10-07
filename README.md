# OpsGenie Pulse

**Cost-aware predictive maintenance with criticality-based decisions, explainable alerts and drift monitoring**
Team26 | Hackathon 4.0 | Track 4: OpsGenie AI | IES MCRC Avenir Analytics Club

Team: Sean Pereira, Pranay Reddy, Ralph Dsouza (B.E. Computer Engineering, Fr. Conceicao Rodrigues College of Engineering)

| | |
|---|---|
| Live demo | `<ADD STREAMLIT LINK>` |
| Demo video (2-3 min) | `<ADD YOUTUBE / DRIVE LINK>` |
| Report | `Team26_OpsGenieAI_Hackathon4.0.pdf` |

<!-- Add a screenshot or GIF here: ![demo](figures/demo.gif) -->

## The idea
Predicting that a machine *might* fail is not the same as deciding whether to service it. OpsGenie Pulse flags a machine when the expected downtime cost avoided is larger than the cost of the job:

`p(failure) x prevented share x failure cost > job cost`

so a critical machine is serviced at a much lower risk score than a low-impact one. The dashboard also monitors drift (PSI) and shows the value of retraining.

## Results (simulated data, 8,000 held-out machine-weeks)
| Policy | Net benefit per 1,000 machine-weeks (INR) |
|---|---|
| Service every machine | -10,180,000 |
| Fixed-interval servicing | -1,176,875 |
| Default 0.5 cut-off | 331,250 |
| Tuned flat cut-off | 1,130,625 |
| **Gradient boosting + expected-value rule** | **1,535,000** |

After drift, retraining lifts net benefit from 2,192,500 to 3,151,500 (+44%). Logistic regression is on par with gradient boosting under the same rule (1,840,625); most of the gain comes from the decision rule.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Optional: `python make_figures.py` saves charts to `figures/`.

## Files
- `opsgenie.py`: simulation, models, decision rule, evaluation, PSI
- `app.py`: Streamlit dashboard (Maintenance simulator, Machine scorer, Drift monitor)
- `make_figures.py`: regenerates charts

## Limitations
All data is simulated; real failures are noisier. The INR 15,000 job cost, 80% prevention share and failure-cost tiers are assumptions to validate in a shadow-mode pilot. Scores support, and do not replace, statutory safety inspections.
