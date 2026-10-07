import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
import opsgenie as og

st.set_page_config(page_title="OpsGenie Pulse", page_icon="🛠️", layout="wide")
G, N, T, RD = "#1f9d8b", "#1f3a5f", "#8a93a6", "#d1495b"

@st.cache_resource
def setup():
    data = og.load_data(); models = og.train(data)
    pv = models["gb"].predict_proba(data["val"][og.FEATURES])[:, 1]
    return data, models, og.best_flat_cutoff(data["val"], pv)
data, models, best_t = setup()
te = data["test"]; pt = models["gb"].predict_proba(te[og.FEATURES])[:, 1]; pl = models["lg"].predict_proba(te[og.FEATURES])[:, 1]

st.title("OpsGenie Pulse")
st.caption("Cost-aware predictive maintenance with criticality-based decisions and drift monitoring | Team26 | Hackathon 4.0 Track 4. All data is simulated.")
tab1, tab2, tab3 = st.tabs(["Maintenance simulator", "Machine scorer", "Drift monitor"])

with st.sidebar:
    st.header("Economics (inputs)")
    job = st.slider("Planned job cost (INR)", 5000, 60000, 15000, 1000)
    prev = st.slider("Share of failures a service prevents", .2, 1.0, .8, .05)
    c1 = st.number_input("Failure cost, low impact (INR)", 10000, 5000000, 50000, 10000)
    c2 = st.number_input("Failure cost, medium impact (INR)", 10000, 5000000, 150000, 10000)
    c3 = st.number_input("Failure cost, critical (INR)", 10000, 5000000, 500000, 10000)
tc = (c1, c2, c3)

with tab1:
    st.subheader("Which policy earns the most?")
    tbl = og.policy_table(te, pt, pl, best_t, job, prev, tc)
    st.dataframe(tbl, hide_index=True)
    best = tbl.iloc[:, 1].idxmax(); st.success(f"Best policy under these inputs: **{tbl.iloc[best, 0]}**")
    st.markdown("**Expected-value rule:** service when `p x prevented share x failure cost > job cost`. Risk bars by tier: " +
                ", ".join(f"{n} {job/(prev*c):.0%}" for n, c in zip(og.TIER_NAMES, tc)) + ".")
    ths, nets = og.net_vs_cutoff(te, pt, job, prev, tc)
    ev_net = og.policy_stats(te, og.ev_flag(te, pt, job, prev, tc), job, prev, tc)["net"]
    fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(ths, nets, color=G, label="Flat cut-off"); ax.axhline(ev_net, ls="--", color=N, label="Expected-value rule")
    ax.axvline(.5, ls=":", color=RD); ax.set_xlabel("Service if failure risk >= cut-off"); ax.set_ylabel("Net benefit per 1,000 (INR)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False); st.pyplot(fig)
    auc, pr = og.model_quality(te, pt); st.caption(f"Gradient boosting on held-out data: ROC-AUC {auc:.2f}, PR-AUC {pr:.2f}.")

with tab2:
    st.subheader("Score a single machine")
    med = data["train"][og.FEATURES].median(); cols = st.columns(5); vals = {}
    rng = {"vibration_rms": (.5, 8., float(med.vibration_rms)), "temp_c": (40., 110., float(med.temp_c)), "pressure_bar": (3., 7., float(med.pressure_bar)),
           "hrs_since_service": (0., 2000., float(med.hrs_since_service)), "load_pct": (10., 100., float(med.load_pct)),
           "oil_quality": (10., 100., float(med.oil_quality)), "age_years": (.2, 25., float(med.age_years)),
           "error_codes_7d": (0., 10., float(round(med.error_codes_7d))), "rpm_dev": (0., 10., float(med.rpm_dev)), "ambient_c": (15., 50., float(med.ambient_c))}
    for i, (f, (lo, hi, dv)) in enumerate(rng.items()):
        vals[f] = cols[i % 5].slider(f, lo, hi, dv)
    tier = st.selectbox("Machine criticality", range(3), format_func=lambda i: f"{og.TIER_NAMES[i]} (failure cost INR {tc[i]:,.0f})", index=1)
    row = pd.DataFrame([vals]); p, drivers = og.explain_machine(models["gb"], row, data["train"])
    bar = job / (prev * tc[tier]); act = p > bar
    a, b = st.columns(2); a.metric("Failure risk (7 days)", f"{p:.1%}"); b.metric("Risk bar for this machine", f"{bar:.1%}")
    (st.error if act else st.success)(f"Decision: **{'SERVICE NOW' if act else 'No service needed'}**")
    st.write("Signals behind the score (change in risk vs a typical machine):")
    for f, dlt in drivers: st.write(f"- `{f}` = {vals[f]:.1f}: {dlt:+.1%}")

with tab3:
    st.subheader("Has the data drifted?")
    pt2 = og.psi_table(data["train"], data["d2"]); st.dataframe(pt2.style.format({"PSI": "{:.2f}"}), hide_index=True)
    major = pt2[pt2.PSI > .25].feature.tolist()
    (st.error if major else st.success)("Major drift in: " + ", ".join(major) + ". Retraining recommended." if major else "No major drift.")
    t2 = data["test2"]; s_p = models["gb"].predict_proba(t2[og.FEATURES])[:, 1]; r_p = models["gb2"].predict_proba(t2[og.FEATURES])[:, 1]
    ss = og.policy_stats(t2, og.ev_flag(t2, s_p, job, prev, tc), job, prev, tc); rs = og.policy_stats(t2, og.ev_flag(t2, r_p, job, prev, tc), job, prev, tc)
    x, y = st.columns(2); x.metric("Stale model, net per 1,000 (INR)", f"{ss['net']:,.0f}", f"{ss['jobs']:.0f} jobs per 1,000", delta_color="off")
    y.metric("Retrained model, net per 1,000 (INR)", f"{rs['net']:,.0f}", f"{rs['net']/ss['net']-1:+.0%} vs stale")
