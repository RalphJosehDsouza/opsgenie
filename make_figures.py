"""Regenerates report-style figures into ./figures"""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import opsgenie as og
os.makedirs("figures", exist_ok=True)
d = og.load_data(); m = og.train(d); te = d["test"]; pt = m["gb"].predict_proba(te[og.FEATURES])[:, 1]
d_ = og.DEFAULTS; ths, nets = og.net_vs_cutoff(te, pt, d_["job_cost"], d_["prevent"], d_["tier_costs"])
ev = og.policy_stats(te, og.ev_flag(te, pt, d_["job_cost"], d_["prevent"]), d_["job_cost"], d_["prevent"])["net"]
plt.figure(figsize=(6, 3)); plt.plot(ths, nets); plt.axhline(ev, ls="--"); plt.xlabel("Cut-off"); plt.ylabel("Net benefit per 1,000 (INR)"); plt.tight_layout(); plt.savefig("figures/net_vs_cutoff.png", dpi=150)
plt.figure(figsize=(6, 3)); plt.bar(range(1, 11), og.decile_failure_rates(te, pt)); plt.xlabel("Risk decile"); plt.ylabel("Failure rate (%)"); plt.tight_layout(); plt.savefig("figures/deciles.png", dpi=150)
t = og.psi_table(d["train"], d["d2"]); plt.figure(figsize=(6, 3)); plt.barh(t.feature[::-1], t.PSI[::-1]); plt.axvline(.25, ls="--"); plt.xlabel("PSI"); plt.tight_layout(); plt.savefig("figures/psi.png", dpi=150)
print("Saved figures/")
