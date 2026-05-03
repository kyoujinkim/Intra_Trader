DATA_DIR          = "./data"
TICK_DIR          = 'D:/TickData_DB'
OUTPUT_DIR        = "./output"
INDICATOR_XLSX    = "./price_microstructure_analysis.xlsx"
META_CSV          = "./output/meta_df.csv"
ANALYSIS_XLSX     = "./output/analysis.xlsx"
RF_DIR            = "./output/RF"
MODEL_PATH        = "./model/rf.joblib"

ROLLING_WINDOW    = 12
MARKET_HOURS_KT      = ("09:30", "15:00")
MARKET_HOURS_   = ("23:50", "07:00")

WINDOW            = 3
STRIDE            = 1

FEATURE_COLS = [
    "str_mean", "str_max", "str_min",
    "res_mean",
    "disp_mean", "disp_max", "disp_min",
    "sd_mean",   "sd_max",  "sd_min",
    "sdd_mean",  "sdd_max", "sdd_min",
]

DATA_FIELD = ['TRDPRC_1', 'BID', 'ASK']

SHEET_NAMES = {
    "strength":    "Strength_Score",
    "resilience":  "Resilience",
    "disparity":   "Disparity",
    "signed_disp": "Signed_Disparity",
    "sdd":         "Signed_Disp_Diff",
    "coil":        "Coil_Release",
    "returns":     "Returns",
}

SIGNAL_CONFIDENCE = 0.60
