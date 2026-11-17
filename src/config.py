DATA_DIR          = "./data"
TICK_DIR          = 'D:/TickData_DB'
OUTPUT_DIR        = "./output"
MODEL_PATH        = "./model"

ROLLING_WINDOW    = 12
MARKET_HOURS      = {'kr':("Asia/Seoul", "09:30", "15:00"),
                     'us':("America/New_York", "09:30", "16:00")}

WINDOW            = 3
STRIDE            = 3

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
    "returns":     "Returns",
}

SIGNAL_CONFIDENCE = 0.60
