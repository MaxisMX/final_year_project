from src.data.fetcher import fetch_ohlcv
from src.features.indicators import add_indicators
from src.features.target import attach_label, drop_unlabelled
from src.models.random_forest import (
    FEATURE_COLUMNS, chronological_split, train_random_forest
)
from src.explain.shap_analysis import compute_shap
from src.explain.shap_english import explain_from_shap
from src.models.confidence import assess_confidence

data = fetch_ohlcv("MSFT")
feat = add_indicators(data)
labelled = attach_label(feat, horizon=5)
clean = drop_unlabelled(labelled, horizon=5).dropna(subset=FEATURE_COLUMNS)

split = chronological_split(clean)
model = train_random_forest(split.X_train, split.y_train)

X = split.X_test.iloc[[-1]]
pred = int(model.predict(X)[0])

shap_df = compute_shap(model, X).shap_values
exp = explain_from_shap(shap_df.iloc[-1], clean.loc[X.index[-1]], pred)

print(exp.to_text())

report = assess_confidence(model, split.X_test)
print(report.status, f"{report.buy_probability:.3f}", f"{report.minority_share:.3f}")
print(report.message)