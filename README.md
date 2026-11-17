# Intra_Trader

한국·미국 주식의 5분 호가·체결가격에서 장중 미시구조 지표를 만들고, **종목의 행동 패턴을 군집화하거나 분류모형으로 분석**하는 Python 리서치 프로젝트입니다.


## 첨부 PDF의 연구 결과

2026-04-06 보고서는 인트라볼 급등 이후의 군집별 수익률과 유동성 위험 신호를 분석합니다. **2025년 3~11월 학습, 2025년 12월~2026년 3월 테스트**로 나누고, 변동성 기준일로부터 2거래일 후 매수하는 설정을 사용했습니다.

- 익일 수익률에서는 상승군과 하락군의 성과가 유사했지만, 향후 5일 평균 수익률에서는 상승군의 강세가 더 오래 유지되고 하락군은 성과가 되돌아가는 경향을 보였습니다.
- KOSPI 200 기준 하락군 종목 비중과 과거 1개월 수익률의 상관계수는 **+0.42**, 향후 1개월 수익률과는 **−0.26**으로 보고됐습니다. 보고서는 매수 우세 종목의 과도한 확산을 유동성 위험 후보 신호로 해석합니다.

이 결과는 해당 기간의 군집·상관 분석입니다. 분류 정확도나 비용 차감 초과수익률을 뜻하지 않으며, 상관관계만으로 인과관계나 미래 예측력을 확정할 수 없습니다. 현재 코드의 국가별 확장 기능 전체를 검증한 성과도 아닙니다.

출처: [260407_인트라볼_팩터.pdf](https://github.com/kyoujinkim/Intra_Trader/blob/master/260407_%EC%9D%B8%ED%8A%B8%EB%9D%BC%EB%B3%BC_%ED%8C%A9%ED%84%B0.pdf) — PDF 18, 19쪽

## 연구 아이디어

같은 수익률을 기록한 종목도 매수·매도 압력과 스프레드, 장중 경로는 다를 수 있습니다. 이 프로젝트는 가격 변동의 결과뿐 아니라 장중 경로의 구조를 특징으로 만듭니다. 기존 연구의 U자형 장중 패턴과 경로 괴리, 낮은 매수 강도에서의 방향성 움직임 가설을 이어갑니다.

```text
Refinitiv/LSEG 5분 BID·ASK·TRDPRC_1
  → 종목별 Parquet
  → 일별 미시구조 지표·변동성 Excel
  → 이동 구간 특징
  → 군집 분석 또는 Random Forest 분류
  → 종목·업종별 결과·그림
```

## 지표와 분석

| 지표·방법 | 해석 |
|---|---|
| 중간가격 `(BID + ASK) / 2` | 호가 중심 가격 |
| 스프레드 `ASK - BID` | 호가 간격 |
| 체결가격의 스프레드 내 위치 | 매수·매도 압력의 근사 |
| strength·resilience | 압력의 지속성과 스프레드의 안정성 |
| disparity·signed disparity·차분 | 기준 장중 경로와 괴리·방향·변화 |
| 변동성·수익률 | 종목·업종별 움직임 요약 |
| K-Means·GMM·계층 군집·DBSCAN | 종목×이동 구간의 행동 패턴 분류 |

지표 이름은 경제적 의미에 대한 연구 가설을 나타냅니다. 호가만으로 투자자 의도나 미래 수익률을 직접 식별하는 것은 아닙니다.

## 현재 구조

| 경로 | 역할 |
|---|---|
| `src/config.py` | 국가별 시간대·분석 시간·데이터 경로 |
| `load_rd.py` | 한국·미국 5분 데이터 조회 |
| `calc_indicator.py` | 지표·변동성 계산 |
| `predict.py` | 특징 생성·분류모형 학습·예측 |
| `src/clustering.py` | 이동 구간별 군집 분석 |
| `src/industry_spotmap.py`, `src/utils.py` | 업종 분석·공통 기능 |
| `data/kr/`, `data/us/` | 국가별 유니버스·업종 매핑 |

기존 README의 ‘군집화 예정’ 설명과 달리 군집화 코드는 `src/clustering.py`에 있습니다. 수집·지표·예측 코드도 현재 `src.config`를 사용합니다.

## 설치·설정

루트에 의존성 파일이 없으므로 다음 주요 패키지의 호환 버전을 별도 환경에서 준비해야 합니다.

```bash
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl pyarrow tqdm python-dateutil refinitiv-data dtaidistance joblib
```

`src/config.py`의 `DATA_DIR`, `TICK_DIR`, `OUTPUT_DIR`, `MODEL_PATH`, `MARKET_HOURS`를 확인합니다. 시간 설정은 코드의 분석 구간이며 거래소의 전체 거래시간과 같다고 가정하지 마세요. 미국은 시간대·일광절약시간을 함께 고려해야 합니다.

조회기는 `D:/config.ini`의 `[main] api_key`를 사용합니다. 위치를 사용자 환경에 맞게 바꾸고 실제 값은 비공개로 유지합니다. 원천 데이터 서비스의 사용 권한이 필요합니다.

국가별 `universe.csv`의 인덱스·`Type`·`RIC`·`Name`은 수집·지표·예측 단계가 읽는 형식에 맞춰야 합니다. 현재 일부 CSV 로더는 `CP949`를 지정합니다. 업종 매핑은 종목 코드와 일치해야 합니다.

## 실행

저장소 루트에서 필요한 경로와 입력을 준비한 뒤 실행합니다.

```bash
python load_rd.py --country kr
python calc_indicator.py --country kr --days 20
python predict.py --country kr --train --start_dt 2026-01-01
```

지표·예측은 `output/<country>/indicators.xlsx` 등을 연결합니다. 예측 실행은 기존 학습 모델의 존재 여부도 확인해야 합니다.

군집 분석은 입력·업종 경로를 명시합니다.

```bash
python src/clustering.py --input output/kr/indicators.xlsx --sector_csv data/kr/sector.csv --window 20 --stride 5 --model gmm --k_range 2 8
```

군집 입력은 날짜×종목 형식의 `Strength_Score`, `Resilience`, `Disparity`, `Signed_Disparity`, `Signed_Disp_Diff`, `Coil_Release`, `Returns` 시트를 읽습니다. 현재 지표 생성기와 해당 시트 구성이 맞는지 확인하고 누락된 지표는 생성·연결해야 합니다. 군집용 업종 파일에는 `Code`, `Sector` 열이 필요합니다.

## 결과와 검증

국가별 지표·변동성 워크북, `output/<country>/result.csv`, 특징 중요도·분류 결과와 군집별 Excel·그림을 생성합니다. 군집 코드 일부는 국가별 출력과 별도로 루트·업종별 경로를 사용하므로 산출물을 구분하세요.

이동 구간의 중첩, 전체 표본으로 적합한 표준화, 평가기간에 포함된 종목·유동성 차이는 검증 결과에 영향을 줍니다. 실시간 전략으로 확장할 때는 학습·예측 기준일을 분리해야 합니다. 기존 문서의 Random Forest 성과에 대한 설명은 연구 동기이며 현재 코드의 재검증 결과가 아닙니다.

## 기존 문서·데이터

[이전 README](README.previous.md)에 U자형 경로와 초기 연구 방향을 보존했습니다. 현재 경로·실행법은 이 문서와 코드를 기준으로 합니다. 원천 분봉·연구 PDF·매핑 자료의 배포·재사용 범위는 각각 확인해야 합니다.
