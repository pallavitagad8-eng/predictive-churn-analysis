"""Predictive Churn Analysis - cleaning, encoding, modelling, evaluation, export."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, roc_curve, confusion_matrix)
import json

RAW = "customer_churn_model_v2.csv"
df = pd.read_csv(RAW)

# ---------- 1. Data cleaning checks ----------
report = {"rows": len(df), "cols": df.shape[1],
          "missing": int(df.isna().sum().sum()),
          "duplicates": int(df.duplicated().sum()),
          "dup_ids": int(df.CustomerID.duplicated().sum())}
for c in df.select_dtypes(exclude="number"):
    df[c] = df[c].str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["Churn_Flag"] = (df["Churn"] == "Yes").astype(int)
df["AutoPay_Flag"] = (df["AutoPay"] == "Yes").astype(int)
df.to_csv("outputs/customer_churn_cleaned.csv", index=False)

# ---------- 2. Feature encoding + scaling ----------
cat_cols = ["Region", "SubscriptionType", "ContractType", "PaymentMethod"]
num_cols = ["Age", "TenureMonths", "MonthlyChargesINR", "TotalChargesINR", "SupportTickets", "LatePayments"]
X = pd.get_dummies(df[num_cols + ["AutoPay_Flag"] + cat_cols], columns=cat_cols, drop_first=True, dtype=int)
y = df["Churn_Flag"]

# ---------- 3. 80/20 stratified split ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler().fit(Xtr[num_cols])          # fit on train only (no leakage)
Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
Xtr_s[num_cols] = scaler.transform(Xtr[num_cols])
Xte_s[num_cols] = scaler.transform(Xte[num_cols])
pd.concat([Xtr_s, ytr], axis=1).to_csv("outputs/train_encoded_scaled.csv", index=False)
pd.concat([Xte_s, yte], axis=1).to_csv("outputs/test_encoded_scaled.csv", index=False)

# ---------- 4. Train models ----------
models = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
          "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)}
res, probs = {}, {}
for n, m in models.items():
    m.fit(Xtr_s, ytr)
    p = m.predict_proba(Xte_s)[:, 1]; pred = (p >= .5).astype(int); probs[n] = p
    res[n] = dict(Accuracy=accuracy_score(yte, pred), Precision=precision_score(yte, pred),
                  Recall=recall_score(yte, pred), F1=f1_score(yte, pred),
                  ROC_AUC=roc_auc_score(yte, p), CM=confusion_matrix(yte, pred).tolist())
best = max(res, key=lambda k: res[k]["ROC_AUC"])

# ---------- 5. Plots ----------
plt.figure(figsize=(5.5, 4.5))
for n, p in probs.items():
    f, t, _ = roc_curve(yte, p); plt.plot(f, t, label=f"{n} (AUC={res[n]['ROC_AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig("outputs/roc_curve.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
for a, n in zip(ax, res):
    cm = np.array(res[n]["CM"]); a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["No", "Yes"]); a.set_yticklabels(["No", "Yes"])
    a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title(n, fontsize=10)
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(figsize=(6, 4)); imp.plot.barh(color="#2f5fa8"); plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()
imp.sort_values(ascending=False).to_csv("outputs/feature_importance.csv", header=["importance"])

# ---------- 6. Export churn risk scores (all customers, best model) ----------
Xall = X.copy(); Xall[num_cols] = scaler.transform(X[num_cols])
score = models[best].predict_proba(Xall)[:, 1]
out = df[["CustomerID", "Churn"]].copy()
out["ChurnProbability"] = score.round(4)
out["RiskLevel"] = pd.cut(score, [-.01, .33, .66, 1.0], labels=["Low", "Medium", "High"])
out["Set"] = np.where(df.index.isin(Xte.index), "Test", "Train")
out.sort_values("ChurnProbability", ascending=False).to_csv("outputs/churn_risk_scores.csv", index=False)

report.update(best=best, results=res, churn_rate=float(y.mean()), n_train=len(Xtr), n_test=len(Xte),
              n_features=X.shape[1], top_features=imp.sort_values(ascending=False).round(4).to_dict(),
              risk_counts=out["RiskLevel"].value_counts().to_dict())
json.dump(report, open("outputs/summary.json", "w"), indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
