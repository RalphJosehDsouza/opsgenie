"""OpsGenie Pulse core logic: simulation, models, decision rule, evaluation, drift (PSI)."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

FEATURES = ["vibration_rms","temp_c","pressure_bar","hrs_since_service","load_pct",
            "oil_quality","age_years","error_codes_7d","rpm_dev","ambient_c"]
TIER_COSTS = (50000.0, 150000.0, 500000.0)   # failure cost: low / medium / critical
TIER_NAMES = ("Low impact", "Medium impact", "Critical")
DEFAULTS = dict(job_cost=15000.0, prevent=0.80, tier_costs=TIER_COSTS)

def gen(n, period, seed, target):
    r = np.random.default_rng(seed)
    vib = np.clip(r.normal(2.5 if period == 1 else 3.1, .8, n), .5, 8)
    temp = r.normal(70, 8, n); pre = r.normal(5, .6, n)
    hrs = np.clip(r.gamma(2, 300, n), 0, 2000)
    load = np.clip(r.normal(65 if period == 1 else 78, 15 if period == 1 else 12, n), 10, 100)
    oil = np.clip(r.beta(6, 2, n) * 100, 10, 100); age = np.clip(r.gamma(3, 3, n), .2, 25)
    err = r.poisson(1.0 if period == 1 else 1.4, n); rpm = np.abs(r.normal(0, 2, n))
    amb = r.normal(28 if period == 1 else 34, 4, n)
    z = 1.7 * (.9*(vib-2.5) + .03*(temp-70) + .0012*hrs + .02*(load-65)*(1 if period == 1 else 2.2)
               - .02*(oil-75) + .04*age + .25*err + .1*rpm + (.03 if period == 1 else .09)*(amb-28)
               + (0 if period == 1 else .6*((load > 85) & (oil < 70))))
    lo, hi = -12, 4
    for _ in range(40):
        m = (lo+hi)/2; pr = 1/(1+np.exp(-(z+m)))
        lo, hi = (m, hi) if pr.mean() < target else (lo, m)
    y = (r.random(n) < pr).astype(int)
    tier = r.choice([0, 1, 2], n, p=[.5, .35, .15])
    return pd.DataFrame(dict(vibration_rms=vib, temp_c=temp, pressure_bar=pre, hrs_since_service=hrs,
        load_pct=load, oil_quality=oil, age_years=age, error_codes_7d=err, rpm_dev=rpm, ambient_c=amb,
        tier=tier, y=y))

def load_data():
    d1 = gen(40000, 1, 1, .04); d2 = gen(20000, 2, 2, .055)
    return dict(train=d1.iloc[:24000], val=d1.iloc[24000:32000], test=d1.iloc[32000:],
                retrain=d2.iloc[:10000], test2=d2.iloc[10000:], d2=d2)

def _gb():
    return HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=.06,
                                          l2_regularization=1.0, random_state=0)

def train(data):
    gb = _gb().fit(data["train"][FEATURES], data["train"].y)
    lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(data["train"][FEATURES], data["train"].y)
    gb2 = _gb().fit(data["retrain"][FEATURES], data["retrain"].y)
    return dict(gb=gb, lg=lg, gb2=gb2)

def fail_cost(d, tier_costs=TIER_COSTS):
    return np.asarray(tier_costs, float)[d.tier.values]

def ev_flag(d, p, job_cost, prevent, tier_costs=TIER_COSTS):
    return p * prevent * fail_cost(d, tier_costs) > job_cost

def net_benefit(d, flag, job_cost, prevent, tier_costs=TIER_COSTS, per=1000):
    v = flag * (d.y.values * prevent * fail_cost(d, tier_costs) - job_cost)
    return v.sum() / len(d) * per

def policy_stats(d, flag, job_cost, prevent, tier_costs=TIER_COSTS):
    y = d.y.values
    return dict(net=net_benefit(d, flag, job_cost, prevent, tier_costs),
                reached=(flag & (y == 1)).sum() / max(y.sum(), 1) * 100,
                jobs=flag.mean() * 1000, unneeded=(flag & (y == 0)).sum() / len(d) * 1000)

def best_flat_cutoff(val, pv, job_cost=15000.0, prevent=.8, tier_costs=TIER_COSTS):
    ths = np.linspace(.02, .9, 89)
    return max(ths, key=lambda t: net_benefit(val, pv >= t, job_cost, prevent, tier_costs))

def policy_table(d, pt, pl, best_t, job_cost, prevent, tier_costs):
    pol = {"Run to failure (no service)": np.zeros(len(d), bool),
           "Service every machine": np.ones(len(d), bool),
           "Fixed-interval servicing (>800 h)": (d.hrs_since_service > 800).values,
           "Gradient boosting, default 0.5 cut-off": pt >= .5,
           f"Gradient boosting, tuned flat cut-off ({best_t:.2f})": pt >= best_t,
           "Logistic regression + expected-value rule": ev_flag(d, pl, job_cost, prevent, tier_costs),
           "Gradient boosting + expected-value rule": ev_flag(d, pt, job_cost, prevent, tier_costs)}
    rows = []
    for k, f in pol.items():
        s = policy_stats(d, f, job_cost, prevent, tier_costs)
        rows.append({"Policy": k, "Net benefit per 1,000 machine-weeks (INR)": round(s["net"]),
                     "Failures reached (%)": round(s["reached"], 1), "Jobs per 1,000": round(s["jobs"]),
                     "Unneeded jobs per 1,000": round(s["unneeded"])})
    return pd.DataFrame(rows)

def net_vs_cutoff(val_or_test, p, job_cost, prevent, tier_costs):
    ths = np.linspace(.02, .9, 89)
    return ths, [net_benefit(val_or_test, p >= t, job_cost, prevent, tier_costs) for t in ths]

def decile_failure_rates(d, p):
    dec = 9 - pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).values
    return [d.y.values[dec == k].mean() * 100 for k in range(10)]

def psi(a, b, bins=10):
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins+1))); e[0], e[-1] = -np.inf, np.inf
    pa = np.histogram(a, e)[0]/len(a) + 1e-4; pb = np.histogram(b, e)[0]/len(b) + 1e-4
    return float(((pb-pa)*np.log(pb/pa)).sum())

def psi_table(train, new):
    t = pd.DataFrame({"feature": FEATURES, "PSI": [psi(train[f].values, new[f].values) for f in FEATURES]})
    t["status"] = np.where(t.PSI > .25, "MAJOR shift", np.where(t.PSI > .10, "watch", "stable"))
    return t.sort_values("PSI", ascending=False).reset_index(drop=True)

def explain_machine(model, row, train, k=3):
    """Local drivers: change in risk when each feature is reset to the training median."""
    base = model.predict_proba(row[FEATURES])[0, 1]; out = []
    for f in FEATURES:
        r2 = row[FEATURES].copy(); r2[f] = train[f].median()
        out.append((f, base - model.predict_proba(r2)[0, 1]))
    return base, sorted(out, key=lambda x: -abs(x[1]))[:k]

def global_importance(model, test):
    pi = permutation_importance(model, test[FEATURES], test.y, scoring="roc_auc", n_repeats=5, random_state=0)
    return sorted(zip(FEATURES, pi.importances_mean), key=lambda x: -x[1])

def model_quality(d, p):
    return roc_auc_score(d.y, p), average_precision_score(d.y, p)
