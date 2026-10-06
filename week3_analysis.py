import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
sns.set_theme(style="whitegrid")
pal = {"Malignant":"#d62728","Benign":"#1f77b4"}
ALPHA = 0.05

# ---------- Data (raw values, no outlier capping: tests should see the real data) ----------
df = load_breast_cancer(as_frame=True).frame
df["diagnosis"] = df["target"].map({0:"Malignant",1:"Benign"}); df = df.drop(columns="target")
df.columns = [c.replace(" ","_") for c in df.columns]
M = df[df.diagnosis=="Malignant"]; Bn = df[df.diagnosis=="Benign"]
print("n malignant", len(M), "n benign", len(Bn))

def welch_ci(a, b, conf=.95):
    diff = a.mean()-b.mean(); va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    se = np.sqrt(va+vb); dof = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
    t = stats.t.ppf(1-(1-conf)/2, dof); return diff, diff-t*se, diff+t*se
def cohen_d(a,b):
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2)); return (a.mean()-b.mean())/sp

# ---------- Test 0: assumptions ----------
print("\n== Assumption checks: mean_radius ==")
for name,g in [("Malignant",M),("Benign",Bn)]:
    w,p = stats.shapiro(g.mean_radius); print(f"Shapiro {name}: W={w:.3f}, p={p:.4f}")
lv = stats.levene(M.mean_radius, Bn.mean_radius); print(f"Levene: stat={lv.statistic:.2f}, p={lv.pvalue:.2e}")

# ---------- Test 1: Welch t-test, mean radius ----------
a,b = M.mean_radius, Bn.mean_radius
t,p = stats.ttest_ind(a,b,equal_var=False); diff,lo,hi = welch_ci(a,b); d = cohen_d(a,b)
print(f"\n== H1 Welch t-test mean_radius ==\nmeans M={a.mean():.2f} B={b.mean():.2f}; t={t:.2f}, p={p:.2e}; diff={diff:.2f} 95%CI[{lo:.2f},{hi:.2f}]; d={d:.2f}")
u,pu = stats.mannwhitneyu(a,b,alternative="two-sided"); print(f"Mann-Whitney U={u:.0f}, p={pu:.2e}")

# ---------- Test 2: non-significant example ----------
a2,b2 = M.mean_fractal_dimension, Bn.mean_fractal_dimension
t2,p2 = stats.ttest_ind(a2,b2,equal_var=False); diff2,lo2,hi2 = welch_ci(a2,b2)
print(f"\n== H2 Welch t-test mean_fractal_dimension ==\nmeans M={a2.mean():.4f} B={b2.mean():.4f}; t={t2:.2f}, p={p2:.3f}; diff={diff2:.5f} 95%CI[{lo2:.5f},{hi2:.5f}]; d={cohen_d(a2,b2):.2f}")

# ---------- Test 3: chi-square size group vs diagnosis ----------
df["size_group"] = pd.qcut(df.mean_area, 4, labels=["Q1 smallest","Q2","Q3","Q4 largest"])
ct = pd.crosstab(df.size_group, df.diagnosis); chi2,pc,dof,exp = stats.chi2_contingency(ct)
cv = np.sqrt(chi2/(ct.values.sum()*(min(ct.shape)-1)))
print(f"\n== H3 Chi-square ==\n{ct}\nchi2={chi2:.1f}, dof={dof}, p={pc:.2e}, Cramer's V={cv:.2f}; min expected={exp.min():.1f}")

# ---------- Test 4: ANOVA across size groups ----------
groups = [g.mean_concave_points.values for _,g in df.groupby("size_group", observed=True)]
F,pa = stats.f_oneway(*groups); H,pk = stats.kruskal(*groups)
grand = df.mean_concave_points.mean(); ssb = sum(len(g)*(g.mean()-grand)**2 for g in groups); sst = ((df.mean_concave_points-grand)**2).sum()
print(f"\n== H4 ANOVA mean_concave_points by size group ==\nF={F:.1f}, p={pa:.2e}, eta^2={ssb/sst:.2f}; Kruskal H={H:.1f}, p={pk:.2e}")
tk = stats.tukey_hsd(*groups); print("Tukey p-values (rows/cols = Q1..Q4):"); print(np.round(tk.pvalue,4))
print("group means:", df.groupby("size_group", observed=True).mean_concave_points.mean().round(3).to_dict())

# ---------- Test 5: all 30 features + multiple-testing correction ----------
rows=[]
for c in [c for c in df.columns if c not in ("diagnosis","size_group")]:
    tt,pp = stats.ttest_ind(M[c],Bn[c],equal_var=False); rows.append((c,tt,pp,cohen_d(M[c],Bn[c])))
res = pd.DataFrame(rows, columns=["feature","t","p","d"])
res["p_bonf"] = np.minimum(res.p*len(res),1); res["p_bh"] = stats.false_discovery_control(res.p.values, method="bh")
print(f"\n== H5 30 t-tests ==\nraw p<0.05: {(res.p<ALPHA).sum()}; Bonferroni: {(res.p_bonf<ALPHA).sum()}; BH-FDR: {(res.p_bh<ALPHA).sum()}")
print("not significant after BH:", res.loc[res.p_bh>=ALPHA,"feature"].tolist())
print(res.reindex(res.d.abs().sort_values(ascending=False).index).head(5).round(3))

# ================= FIGURES =================
fig,ax = plt.subplots(1,2, figsize=(11,4.2))
for diag,g in [("Benign",Bn),("Malignant",M)]:
    sns.histplot(g.mean_radius, kde=True, color=pal[diag], alpha=.5, ax=ax[0], label=f"{diag} (mean {g.mean_radius.mean():.1f})")
    ax[0].axvline(g.mean_radius.mean(), color=pal[diag], ls="--")
ax[0].set_title("Mean radius: two clearly different groups"); ax[0].set_xlabel("mean radius"); ax[0].legend()
for i,(diag,g) in enumerate([("Benign",Bn),("Malignant",M)]):
    m=g.mean_radius.mean(); h=stats.t.ppf(.975,len(g)-1)*g.mean_radius.sem()
    ax[1].errorbar(i, m, yerr=h, fmt="o", color=pal[diag], capsize=8, markersize=9)
    ax[1].text(i+0.08, m, f"{m:.2f}\n±{h:.2f}", va="center")
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["Benign","Malignant"]); ax[1].set_xlim(-.5,1.7)
ax[1].set_ylabel("mean radius (95% CI)"); ax[1].set_title(f"Welch t-test: t = {t:.1f}, p < 0.001")
plt.tight_layout(); plt.savefig("w3figs/s1_hist_ci.png", dpi=150); plt.close()

fig,ax = plt.subplots(1,2, figsize=(9,4))
for a_,(diag,g) in zip(ax,[("Benign",Bn),("Malignant",M)]):
    stats.probplot(g.mean_radius, dist="norm", plot=a_); a_.set_title(f"Q-Q plot: {diag} mean radius")
    a_.get_lines()[0].set_color(pal[diag]); a_.get_lines()[0].set_markersize(3)
plt.tight_layout(); plt.savefig("w3figs/s2_qq.png", dpi=150); plt.close()

feats = ["mean_radius","mean_texture","mean_smoothness","mean_concavity","mean_symmetry","mean_fractal_dimension"]
fig,ax = plt.subplots(figsize=(7.5,4))
for i,f in enumerate(feats):
    x,y = M[f],Bn[f]; dd = cohen_d(x,y); n1,n2=len(x),len(y)
    se = np.sqrt((n1+n2)/(n1*n2)+dd**2/(2*(n1+n2))); sigf = abs(dd)-1.96*se>0
    ax.errorbar(dd, i, xerr=1.96*se, fmt="o", color="#d62728" if sigf else "grey", capsize=4)
    ax.text(dd, i-0.28, f"d={dd:.2f}", ha="center", fontsize=8)
ax.axvline(0,c="k",lw=.8); ax.set_yticks(range(len(feats))); ax.set_yticklabels([f.replace("_"," ") for f in feats]); ax.invert_yaxis()
ax.set_xlabel("Effect size (Cohen's d) with 95% CI"); ax.set_title("How big is the difference? (malignant vs benign)", fontweight="bold")
plt.tight_layout(); plt.savefig("w3figs/s3_forest.png", dpi=150); plt.close()

fig,ax = plt.subplots(1,2, figsize=(11,4.2))
pct = ct.div(ct.sum(axis=1), axis=0)*100
pct[["Benign","Malignant"]].plot(kind="bar", stacked=True, color=[pal["Benign"],pal["Malignant"]], ax=ax[0], rot=0)
ax[0].set_ylabel("% of patients"); ax[0].set_xlabel("Tumour size group"); ax[0].set_title("Share of each diagnosis by size group"); ax[0].legend(title="")
resid = (ct.values-exp)/np.sqrt(exp)
sns.heatmap(pd.DataFrame(resid, index=ct.index, columns=ct.columns), annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax[1], cbar_kws={"label":"standardised residual"})
ax[1].set_title(f"Chi-square residuals (χ²={chi2:.0f}, p < 0.001)"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig("w3figs/s4_chi2.png", dpi=150); plt.close()

plt.figure(figsize=(7,4.5))
sns.boxplot(data=df, x="size_group", y="mean_concave_points", hue="size_group", palette="Reds", legend=False)
plt.title(f"ANOVA: concave points rise with tumour size (F = {F:.0f}, η² = {ssb/sst:.2f})", fontsize=10, fontweight="bold")
plt.xlabel("Tumour size group (mean area quartile)"); plt.tight_layout(); plt.savefig("w3figs/s5_anova.png", dpi=150); plt.close()

plt.figure(figsize=(7.5,5))
sig = res.p_bh<ALPHA
plt.scatter(res.d[sig], -np.log10(res.p[sig]), c="#d62728", label="significant after FDR correction", s=40)
plt.scatter(res.d[~sig], -np.log10(res.p[~sig]), c="grey", label="not significant", s=40)
plt.axhline(-np.log10(ALPHA), ls="--", c="k", lw=.8)
for _,r in res.sort_values("d",key=abs,ascending=False).head(3).iterrows(): plt.text(r.d, -np.log10(r.p)+.4, r.feature.replace("_"," "), fontsize=7, ha="right")
for _,r in res[~sig].iterrows(): plt.text(r.d, -np.log10(r.p)+.4, r.feature.replace("_"," "), fontsize=7, ha="center")
plt.xlabel("Effect size (Cohen's d)"); plt.ylabel("-log10(p-value)"); plt.legend(loc="upper left", fontsize=8)
plt.title("30 feature tests: size of effect vs strength of evidence", fontweight="bold"); plt.tight_layout(); plt.savefig("w3figs/s6_volcano.png", dpi=150); plt.close()
res.round(4).to_csv("w3figs/all_feature_tests.csv", index=False)
