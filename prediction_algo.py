import argparse

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from config import WINDOW, STRIDE, META_CSV, RF_DIR, MODEL_PATH
from utils import setup_matplotlib, safe_show, load_indicator_excel, window_features

setup_matplotlib()


def load_and_preprocess(start_dt=None, window=WINDOW, stride=STRIDE):
    print("[1/5] Loading data from Excel...")
    sheets = load_indicator_excel(start_dt=start_dt)
    for key, df in sheets.items():
        print(f"    {key}: {df.shape[0]} dates × {df.shape[1]} tickers")

    common_dates = list(sheets.values())[0].index
    for df in sheets.values():
        common_dates = common_dates.intersection(df.index)
    common_dates = common_dates.sort_values()

    tickers = sheets["strength"].columns.tolist()
    feat_records, y_records, meta_records = [], [], []

    for ticker in tqdm(tickers):
        s   = sheets["strength"][ticker].reindex(common_dates)
        res = sheets["resilience"][ticker].reindex(common_dates)
        d   = sheets["disparity"][ticker].reindex(common_dates)
        sd  = sheets["signed_disp"][ticker].reindex(common_dates)
        sdd = sheets["sdd"][ticker].reindex(common_dates)
        ret = sheets["returns"][ticker].reindex(common_dates)

        n = len(common_dates)
        for start in range(0, n - window + 1, stride):
            end = start + window
            ws = {
                "s":   s.iloc[start:end].dropna(),
                "res": res.iloc[start:end].dropna(),
                "d":   d.iloc[start:end].dropna(),
                "sd":  sd.iloc[start:end].dropna(),
                "sdd": sdd.iloc[start:end].dropna(),
            }
            t_ret = ret.iloc[end:end + 3].dropna()

            if any(len(ws[k]) < window // 2 for k in ws):
                continue

            feat_records.append(window_features(ws["s"], ws["res"], ws["d"], ws["sd"], ws["sdd"]))
            y_records.append({
                "ret_mean":   t_ret.mean(),
                "ret_std":    t_ret.std(),
                "ret_sharpe": t_ret.mean() / (t_ret.std() + 1e-9),
            })
            meta_records.append({
                "ticker":       ticker,
                "window_start": common_dates[start],
                "window_end":   common_dates[end - 1],
            })

    feat_df = pd.DataFrame(feat_records)
    y_df    = pd.DataFrame(y_records)
    meta_df = pd.DataFrame(meta_records)

    y_df['target'] = y_df["ret_mean"].apply(lambda x: 1 if x > 0.01 else -1 if x < -0.01 else 0)

    valid   = feat_df.notna().all(axis=1)
    feat_df = feat_df[valid].reset_index(drop=True)
    y_df    = y_df[valid].reset_index(drop=True)
    meta_df = meta_df[valid].reset_index(drop=True)

    meta_df = meta_df.sort_values(by='window_end')
    feat_df = feat_df.loc[meta_df.index]
    y_df    = y_df.loc[meta_df.index]

    meta_df = meta_df.reset_index(drop=True)
    feat_df = feat_df.reset_index(drop=True)
    y_df    = y_df.reset_index(drop=True)

    print(f"    Windowed samples : {len(feat_df)}  "
          f"({len(tickers)} tickers × ~{len(feat_df) // max(len(tickers), 1)} windows each)")
    print(f"    Feature columns  : {feat_df.shape[1]}")
    return feat_df, y_df, meta_df


def train_model(X, Y, shuffle=True):
    X = X.copy()
    y = Y['target']

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )

    print("--- 모델 학습 및 검증 시작 ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=shuffle
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"검증 정확도: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {classification_report(y_test, preds, digits=4)}")

    y_real = Y['ret_mean']
    analyze_model_results(model, X_test, y_test, y_real, features=X.columns.tolist())

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("\n--- 특성 중요도 (Feature Importance) ---")
    print(feat_imp)

    joblib.dump(model, MODEL_PATH)
    return model


def analyze_model_results(model, X_test, y_test, y_real, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    importance_df = pd.DataFrame({
        'Feature':    features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(f"{RF_DIR}/feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.title('지표별 예측 기여도 (Feature Importance)')
    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('상대적 중요도')
    plt.tight_layout()

    y_probs = model.predict_proba(X_test)[:, 1]
    pd.DataFrame(y_probs, columns=['Up_Prob']).to_csv(f"{RF_DIR}/prediction_confidence.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.histplot(y_probs, bins=50, kde=True, color='purple')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.title('모델의 예측 확신도(Probability) 분포')
    plt.xlabel('상승 확률 (0.5 이상이면 상승 예측)')
    plt.ylabel('종목 수')

    y_preds = model.predict(X_test)
    cm = confusion_matrix(y_test, y_preds)
    pd.DataFrame(cm).to_csv(f"{RF_DIR}/confusion_matrix.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('예측 정확도 매트릭스 (Confusion Matrix)')
    plt.ylabel('실제 결과 (Down/Up)')
    plt.xlabel('모델의 예측 (Down/Up)')

    X_test['Return'] = y_real
    X_test['prediction'] = y_preds
    features_per_cluster = X_test.groupby('prediction').median()
    radar_data = features_per_cluster.copy()
    radar_norm = (radar_data - radar_data.mean()) / (radar_data.std() + 1e-9)
    pd.concat([features_per_cluster, radar_norm], axis=1).to_csv(f"{RF_DIR}/radar_total.csv")

    return y_probs


def coil_release_deep_dive(df, ticker_name):
    stock_df = df[df['ticker'] == ticker_name].tail(60)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(stock_df['ymd'], stock_df['returns'].cumprod(), label='Price Path', color='black', linewidth=2)
    ax1.set_ylabel('Price (Cumulative Returns)')

    ax2 = ax1.twinx()
    ax2.fill_between(stock_df['ymd'], stock_df['cr_rolling_5'], 0,
                     where=(stock_df['cr_rolling_5'] < 0), color='blue', alpha=0.3, label='Coiling (-)')
    ax2.fill_between(stock_df['ymd'], stock_df['cr_rolling_5'], 0,
                     where=(stock_df['cr_rolling_5'] > 0), color='red', alpha=0.3, label='Release (+)')
    ax2.set_ylabel('Coil-Release Intensity')

    plt.title(f'종목 분석: {ticker_name} - Coil-Release 패턴과 가격 동행성')
    fig.tight_layout()
    safe_show()


def load_model(path=MODEL_PATH):
    return joblib.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='주가 예측 알고리즘 학습 및 분석')
    parser.add_argument('--window',   type=int, default=WINDOW, help='특성 추출을 위한 윈도우 크기')
    parser.add_argument('--stride',   type=int, default=STRIDE, help='윈도우 이동 간격')
    parser.add_argument('--train',    action='store_true',       help='모델 학습 여부')
    parser.add_argument('--start_dt', type=str, default='2026-01-01', help='예측 시작 날짜')
    args = parser.parse_args()

    if args.train:
        feat_df, y_df, _ = load_and_preprocess(window=args.window, stride=args.stride)
        trained_rf = train_model(feat_df, y_df, shuffle=False)
    else:
        trained_rf = load_model()

    feat_df, y_df, meta_df = load_and_preprocess(start_dt=args.start_dt, window=args.window, stride=1)

    preds  = trained_rf.predict(feat_df)
    probas = trained_rf.predict_proba(feat_df)
    classes = trained_rf.classes_.tolist()

    meta_df['code']      = meta_df['ticker'].apply(lambda x: 'A' + x[3:9])
    meta_df['preds']     = preds
    meta_df['proba_1']   = probas[:, classes.index(1)]   if  1 in classes else 0.0
    meta_df['proba_neg1']= probas[:, classes.index(-1)]  if -1 in classes else 0.0
    meta_df = pd.concat([meta_df, feat_df], axis=1)

    meta_df.to_csv(META_CSV, index=False)

    print("\n[성공] 알고리즘 설계 및 학습이 완료되었습니다.")
    print("이제 이 모델을 사용하여 장 마감 후 'cr_rolling_5'가 마이너스 극값인 종목을 스캔할 수 있습니다.")
