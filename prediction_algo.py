import argparse

import joblib
import matplotlib
import pandas as pd
import numpy as np
from openpyxl.utils.units import cm_to_dxa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.font_manager as fm
from tqdm import tqdm

from clustering import plot_cluster_radar


OUTPUT_DIR = "./output/RF"


# Tkinter RuntimeError 방지 및 백엔드 최적화
def setup_matplotlib_backend():
    """
    환경에 따라 가장 적합한 백엔드를 설정합니다.
    대화형 백엔드(TkAgg 등)를 먼저 시도하고, 실패 시에만 비대화형(Agg)으로 전환합니다.
    """
    current_backend = matplotlib.get_backend()

    # 이미 백엔드가 설정되어 있고 RuntimeError가 없다면 유지
    try:
        # Pycharm/Spyder 등에서 발생할 수 있는 GUI 스레드 이슈 방지용 테스트
        plt.figure().clear()
        plt.close()
    except (RuntimeError, ImportError, Exception):
        try:
            # 안전한 대화형 백엔드 시도
            matplotlib.use('TkAgg')
        except:
            # 최종적으로 비대화형 백엔드로 폴백 (이 경우 plt.show()는 동작하지 않음)
            matplotlib.use('Agg')
            print("[시스템] 비대화형 백엔드(Agg)가 활성화되었습니다. 그림은 파일로 저장하거나 IDE의 SciView를 확인하세요.")


setup_matplotlib_backend()

def safe_show():
    """
    현재 백엔드가 대화형인 경우에만 plt.show()를 호출하여 경고를 방지합니다.
    """
    backend = matplotlib.get_backend().lower()
    interactive_backends = ['tkagg', 'qt5agg', 'macosx', 'nbagg', 'module://ipykernel.pylab.backend_inline']
    if any(ib in backend for ib in interactive_backends):
        plt.show()
    else:
        # 비대화형일 경우 마지막 생성된 그림을 파일로 저장하거나 안내
        pass

# 시각화 한글 깨짐 방지 및 폰트 설정
def set_korean_font():
    # 시스템에 설치된 폰트 중 한글 지원 폰트 탐색
    font_list = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans CJK KR', 'DejaVu Sans']
    found = False

    # fm.font_manager.ttflist 대신 fm.fontManager.ttflist를 사용해야 합니다.
    for font in font_list:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font
            found = True
            break

    # 한글 폰트를 찾지 못한 경우 기본 폰트로 유지하되 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    return found

# 폰트 설정 실행
if not set_korean_font():
    print("[경고] 한글 폰트를 찾을 수 없습니다. 시각화 시 한글이 깨질 수 있습니다.")

def _window_features(s, res, d, sd, sdd) -> dict:
    """
    Extract statistical features from one window slice of all indicators.
    Returns a flat dict. Called once per (ticker, window).
    """
    row = {}

    # ── Strength ───────────────────────────────────────────────
    row["str_mean"]      = s.mean()
    row["str_max"]       = s.max()
    row["str_min"]       = s.min()

    # ── Resilience ─────────────────────────────────────────────
    row["res_mean"]      = res.mean()

    # ── Disparity ──────────────────────────────────────────────
    row["disp_mean"]     = d.mean()
    row["disp_max"]      = d.max()
    row["disp_min"]      = d.min()

    # ── Signed_Disparity ──────────────────────────────────────────────
    row["sd_mean"]       = sd.mean()
    row["sd_max"]        = sd.max()
    row["sd_min"]        = sd.min()

    # ── Signed_Disp_Diff ──────────────────────────────────────────────
    row["sdd_mean"]      = sdd.mean()
    row["sdd_max"]       = sdd.max()
    row["sdd_min"]       = sdd.min()

    return row


def load_and_preprocess(start_dt=None, window=3, stride=3):
    # 1. 데이터 로드 (사용자가 업로드한 파일명 기준)
    sheet_map = {
        'strength': 'Strength_Score',
        'resilience': 'Resilience',
        'disparity': 'Disparity',
        'signed_disp': 'Signed_Disparity',
        's_disp_diff': 'Signed_Disp_Diff',
        'coil_release': 'Coil_Release',
        'returns': 'Returns'
    }

    sheets = {}
    print("[1/5] Loading data from Excel...")
    for key, sheet in sheet_map.items():
        df = pd.read_excel('price_microstructure_analysis.xlsx', sheet_name=sheet, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if start_dt is not None:
            df = df[df.index >= pd.to_datetime(start_dt)]
        sheets[key] = df
        print(f"    {key}: {df.shape[0]} dates × {df.shape[1]} tickers")

    # Align all sheets on a common date index
    common_dates = sheets["strength"].index
    for key in sheets:
        common_dates = common_dates.intersection(sheets[key].index)
    common_dates = common_dates.sort_values()

    tickers = sheets["strength"].columns.tolist()
    feat_records, y_records, meta_records = [], [], []

    for ticker in tqdm(tickers):
        # Extract aligned series for this ticker
        cols = {
            "s": sheets["strength"][ticker].reindex(common_dates),
            "res": sheets["resilience"][ticker].reindex(common_dates),
            "d": sheets["disparity"][ticker].reindex(common_dates),
            "sd": sheets["signed_disp"][ticker].reindex(common_dates),
            "sdd": sheets["s_disp_diff"][ticker].reindex(common_dates),
            "cr": sheets["coil_release"][ticker].reindex(common_dates),
        }
        col_target = {
            "ret": sheets["returns"][ticker].reindex(common_dates),
        }

        n = len(common_dates)
        starts = range(0, n - window + 1, stride)

        for start in starts:
            end = start + window
            w = {k: v.iloc[start:end].dropna() for k, v in cols.items()}
            t = {k: v.iloc[end:end+3].dropna() for k, v in col_target.items()}

            # Skip windows with too much missing data
            if any(len(w[k]) < window // 2 for k in w):
                continue

            row = _window_features(w["s"], w["res"], w["d"], w["sd"], w["sdd"])
            feat_records.append(row)

            y_records.append({
                "ret_mean": t["ret"].mean(),
                "ret_std": t["ret"].std(),
                "ret_sharpe": t["ret"].mean() / (t["ret"].std() + 1e-9),
            })

            meta_records.append({
                "ticker": ticker,
                "window_start": common_dates[start],
                "window_end": common_dates[end - 1],
            })

    feat_df = pd.DataFrame(feat_records)
    y_df = pd.DataFrame(y_records)
    meta_df = pd.DataFrame(meta_records)

    y_df['target'] = y_df["ret_mean"].apply(lambda x: 1 if x > 0.01 else -1 if x < -0.01 else 0)
    # Drop rows with any NaN features
    valid = feat_df.notna().all(axis=1)
    feat_df = feat_df[valid].reset_index(drop=True)
    y_df = y_df[valid].reset_index(drop=True)
    meta_df = meta_df[valid].reset_index(drop=True)

    # re sort by meta_df's window_end
    meta_df = meta_df.sort_values(by='window_end')
    feat_df = feat_df.loc[meta_df.index]
    y_df = y_df.loc[meta_df.index]

    # reset index
    meta_df = meta_df.reset_index(drop=True)
    feat_df = feat_df.reset_index(drop=True)
    y_df = y_df.reset_index(drop=True)

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
        class_weight='balanced_subsample'  # 데이터 불균형 해소
    )

    print("--- 모델 학습 및 검증 시작 ---")

    '''# 시계열 교차 검증 (Time Series Split)
    tscv = TimeSeriesSplit(n_splits=2)
    
    for train_index, test_index in tscv.split(X):
        cv_X_train, cv_X_test = X.iloc[train_index], X.iloc[test_index]
        cv_y_train, cv_y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(cv_X_train, cv_y_train)
        preds = model.predict(cv_X_test)

        print(f"검증 정확도: {accuracy_score(cv_y_test, preds):.4f}")
        print(f"F1 Score: {classification_report(cv_y_test, preds, digits=4)}")'''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=shuffle)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"검증 정확도: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {classification_report(y_test, preds, digits=4)}")

    y_real = Y['ret_mean']
    analyze_model_results(model, X_test, y_test, y_real, features=X.columns.tolist())

    # 최종 특성 중요도 출력
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    print("\n--- 특성 중요도 (Feature Importance) ---")
    print(feat_imp)

    # Save Model
    joblib.dump(model, "./model/rf.joblib")

    return model


def analyze_model_results(model, X_test, y_test, y_real, features):
    """
    학습된 모델의 결과를 심층 분석하고 시각화합니다.
    """
    # 1. 특성 중요도 (Feature Importance)
    importances = model.feature_importances_
    indices = np.argsort(importances)

    # 1-1. Save Feature Importance into OUTPUT_DIR
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.title('지표별 예측 기여도 (Feature Importance)')
    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('상대적 중요도')
    plt.tight_layout()
    #safe_show()

    # 2. 예측 확률 분포 (Confidence Analysis)
    y_probs = model.predict_proba(X_test)[:, 1]

    # 2-1. Save Confidence Analysis
    y_probs_df = pd.DataFrame(y_probs, columns=['Up_Prob'])
    y_probs_df.to_csv(f"{OUTPUT_DIR}/prediction_confidence.csv", index=False)

    plt.figure(figsize=(10, 5))
    sns.histplot(y_probs, bins=50, kde=True, color='purple')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.title('모델의 예측 확신도(Probability) 분포')
    plt.xlabel('상승 확률 (0.5 이상이면 상승 예측)')
    plt.ylabel('종목 수')
    #safe_show()

    # 3. 혼동 행렬 (Confusion Matrix)
    y_preds = model.predict(X_test)
    cm = confusion_matrix(y_test, y_preds)

    # 3-1. Save Confusion Matrix
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('예측 정확도 매트릭스 (Confusion Matrix)')
    plt.ylabel('실제 결과 (Down/Up)')
    plt.xlabel('모델의 예측 (Down/Up)')
    #safe_show()

    # 4. 예측군별 feature값 평균
    X_test['Return'] = y_real
    X_test['prediction'] = y_preds
    features_per_cluster = X_test.groupby('prediction').median()
    # Radar + heatmap
    radar_data = features_per_cluster.copy()
    radar_norm = (radar_data - radar_data.mean()) / (radar_data.std() + 1e-9)

    # 4-1. Save radar_data and radar_norm
    radar_total = pd.concat([features_per_cluster, radar_norm], axis=1)
    radar_total.to_csv(f"{OUTPUT_DIR}/radar_total.csv")

    #plot_cluster_radar(radar_norm, f"{OUTPUT_DIR}/radar_cluster.png")
    #plot_cluster_heatmap(radar_data, f"{tag}/04b_heatmap_{args.model}.png")
    #print(f"    Radar/Heatmap → ./output/{tag}/04_radar_*.png")

    return y_probs


def coil_release_deep_dive(df, ticker_name):
    """
    특정 종목에 대해 사용자님의 CR 이론이 어떻게 작동했는지 심층 시각화합니다.
    """
    stock_df = df[df['ticker'] == ticker_name].tail(60)  # 최근 60일 데이터

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 주가(수익률 누적) 시각화
    ax1.plot(stock_df['ymd'], stock_df['returns'].cumprod(), label='Price Path', color='black', linewidth=2)
    ax1.set_ylabel('Price (Cumulative Returns)')

    # Coil-Release Rolling 지표 시각화
    ax2 = ax1.twinx()
    ax2.fill_between(stock_df['ymd'], stock_df['cr_rolling_5'], 0,
                     where=(stock_df['cr_rolling_5'] < 0), color='blue', alpha=0.3, label='Coiling (-)')
    ax2.fill_between(stock_df['ymd'], stock_df['cr_rolling_5'], 0,
                     where=(stock_df['cr_rolling_5'] > 0), color='red', alpha=0.3, label='Release (+)')
    ax2.set_ylabel('Coil-Release Intensity')

    plt.title(f'종목 분석: {ticker_name} - Coil-Release 패턴과 가격 동행성')
    fig.tight_layout()
    safe_show()


def load_model(path = './model/rf.joblib'):
    model = joblib.load(path)

    return model


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description='주가 예측 알고리즘 학습 및 분석')
    parser.add_argument('--window', type=int, default=3, help='특성 추출을 위한 윈도우 크기 (기본: 3)')
    parser.add_argument('--stride', type=int, default=3, help='윈도우 이동 간격 (기본: 3)')
    parser.add_argument('--train', action='store_true', help='모델 학습 여부 (기본: False)')
    parser.add_argument('--start_dt', type=str, default='2026-01-01', help='예측 시작 날짜 (예: 2025-01-01, 기본: 전체 데이터)')
    args = parser.parse_args()

    if args.train:
        feat_df, y_df, _ = load_and_preprocess(window=args.window, stride=args.stride)
        trained_rf = train_model(feat_df, y_df, shuffle=False)
    else:
        trained_rf = load_model()

    feat_df, y_df, meta_df = load_and_preprocess(start_dt=args.start_dt, window=args.window, stride=1)

    preds = trained_rf.predict(feat_df)

    meta_df['code'] = meta_df['ticker'].apply(lambda x: 'A' + x[3:9])
    meta_df['preds'] = preds
    meta_df = pd.concat([meta_df, feat_df], axis=1)

    meta_df.to_csv('./output/meta_df.csv', index=False)

    print("\n[성공] 알고리즘 설계 및 학습이 완료되었습니다.")
    print("이제 이 모델을 사용하여 장 마감 후 'cr_rolling_5'가 마이너스 극값인 종목을 스캔할 수 있습니다.")


