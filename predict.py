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

from src.config import WINDOW, STRIDE, MODEL_PATH, SHEET_NAMES, OUTPUT_DIR, DATA_DIR
from src.utils import setup_matplotlib, window_features

setup_matplotlib()


class Predictor:
    def __init__(self, country):
        self.country = country
        self.model = None

    def load_indicator_excel(self, start_dt=None):
        """Load all indicator sheets from the Excel file.

        Returns a dict keyed by config.SHEET_NAMES keys (e.g. 'strength', 'resilience', ...).
        """
        sheets = {}
        for key, sheet_name in SHEET_NAMES.items():
            df = pd.read_excel(f"{OUTPUT_DIR}/{self.country}/indicators.xlsx", sheet_name=sheet_name, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if start_dt is not None:
                df = df[df.index >= pd.to_datetime(start_dt)]
            sheets[key] = df
        return sheets

    def load_data(self, start_dt=None, window=WINDOW, stride=STRIDE):
        print("[1/5] Loading data from Excel...")
        sheets = self.load_indicator_excel(start_dt=start_dt)
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

        y_df['target'] = y_df["ret_mean"].apply(lambda x: 1 if x > 0.03 else -1 if x < -0.01 else 0)

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

        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"    Windowed samples : {len(feat_df)}  "
              f"({len(tickers)} tickers × ~{len(feat_df) // max(len(tickers), 1)} windows each)")
        print(f"    Feature columns  : {feat_df.shape[1]}")
        return feat_df, y_df, meta_df


    def train(self, X, Y, shuffle=True):
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
        self.analyze_model_results(model, X_test, y_test, y_real, features=X.columns.tolist())

        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        print("\n--- 특성 중요도 (Feature Importance) ---")
        print(feat_imp)

        joblib.dump(model, f"{MODEL_PATH}/{self.country}/rf.joblib")
        self.model = model

        return True

    def predict(self, X):
        if self.model is None:
            self.model = self.load_model()
        return self.model.predict(X), self.model.predict_proba(X)

    def analyze_model_results(self, model, X_test, y_test, y_real, features):
        importances = model.feature_importances_
        indices = np.argsort(importances)

        importance_df = pd.DataFrame({
            'Feature':    features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        importance_df.to_csv(f"{OUTPUT_DIR}/{self.country}/feature_importance.csv", index=False)

        plt.figure(figsize=(10, 6))
        plt.title('지표별 예측 기여도 (Feature Importance)')
        plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('상대적 중요도')
        plt.tight_layout()

        y_probs = model.predict_proba(X_test)[:, 1]
        pd.DataFrame(y_probs, columns=['Up_Prob']).to_csv(f"{OUTPUT_DIR}/{self.country}/prediction_confidence.csv", index=False)

        plt.figure(figsize=(10, 5))
        sns.histplot(y_probs, bins=50, kde=True, color='purple')
        plt.axvline(0.5, color='red', linestyle='--')
        plt.title('모델의 예측 확신도(Probability) 분포')
        plt.xlabel('상승 확률 (0.5 이상이면 상승 예측)')
        plt.ylabel('종목 수')

        y_preds = model.predict(X_test)
        cm = confusion_matrix(y_test, y_preds)
        pd.DataFrame(cm).to_csv(f"{OUTPUT_DIR}/{self.country}/confusion_matrix.csv")

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
        pd.concat([features_per_cluster, radar_norm], axis=1).to_csv(f"{OUTPUT_DIR}/{self.country}/radar_total.csv")

        return y_probs

    def load_model(self):
        return joblib.load(f"{MODEL_PATH}/{self.country}/rf.joblib")

def parse_args():
    parser = argparse.ArgumentParser(description='주가 예측 알고리즘 학습 및 분석')
    parser.add_argument('--window',   type=int, default=WINDOW, help='특성 추출을 위한 윈도우 크기')
    parser.add_argument('--stride',   type=int, default=STRIDE, help='윈도우 이동 간격')
    parser.add_argument('--train',    action='store_true',       help='모델 학습 여부')
    parser.add_argument('--start_dt', type=str, default='2026-01-01', help='예측 시작 날짜')
    parser.add_argument('-c', '--country', type=str, default='us', help='대상 국가 (예: kr, us)', choices=['kr', 'us'])
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    predictor = Predictor(country=args.country)

    if args.train:
        feat_df, y_df, _ = predictor.load_data(window=args.window, stride=args.stride)
        predictor.train(feat_df, y_df, shuffle=False)

    feat_df, y_df, meta_df = predictor.load_data(start_dt=args.start_dt, window=args.window, stride=1)

    preds, probas = predictor.predict(feat_df)
    classes = predictor.model.classes_.tolist()

    universe = pd.read_csv(f"{DATA_DIR}/{args.country}/universe.csv", encoding='CP949', index_col=0)

    meta_df['code']      = meta_df['ticker'].apply(lambda x: universe.loc[x].RIC)
    meta_df['name']      = meta_df['ticker'].apply(lambda x: universe.loc[x].Name)
    meta_df['preds']     = preds
    meta_df['proba_1']   = probas[:, classes.index(1)]   if  1 in classes else 0.0
    meta_df['proba_neg1']= probas[:, classes.index(-1)]  if -1 in classes else 0.0
    meta_df = pd.concat([meta_df, feat_df], axis=1)

    meta_df.to_csv(f"{OUTPUT_DIR}/{args.country}/result.csv", index=False, encoding='CP949')

    print("\n[성공] 알고리즘 설계 및 학습이 완료되었습니다.")
    print("이제 이 모델을 사용하여 장 마감 후 'cr_rolling_5'가 마이너스 극값인 종목을 스캔할 수 있습니다.")
