from __future__ import annotations
import MetaTrader5 as mt5 # import thư viện của mt5
# file: get_eurusd.py
import datetime as dt
import pandas as pd# thư viện xử lý bảng
import MetaTrader5 as mt5
from typing import Optional, Tuple
from typing import Optional, Dict   # type hint cho tham số
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

#from features import make_features
# file: io.py

def load_clean_csv(
    path: str,
    time_col: str = "time",
    cols_map: dict | None = None,
    usecols: list[str] | None = None,
    tz: str | None = None,
    resample_rule: str | None = None,
    resample_ohlc: bool = True,
    fill_method: str = "ffill",
    dropna_all: bool = True,
    dedup_on: str = "time",
):
    # 1) Đọc CSV: parse cột thời gian thành datetime (không cần infer_datetime_format)
    df = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=[time_col],   # đủ rồi
    )

    # 2) Chuẩn hóa tên cột
    df.columns = [c.lower() for c in df.columns]
    if cols_map:
        normalized = {k.lower(): v.lower() for k, v in cols_map.items()}
        df = df.rename(columns=normalized)
    if time_col.lower() != "time":
        df = df.rename(columns={time_col.lower(): "time"})

    # 3) Sort & dedup
    df = df.sort_values("time")
    if dedup_on:
        df = df.drop_duplicates(subset=[dedup_on])

    # 4) Gắn/đổi timezone (nếu muốn)
    if tz is not None:
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(tz)
        else:
            df["time"] = df["time"].dt.tz_convert(tz)

    # 5) Bỏ dòng toàn NA
    if dropna_all:
        df = df.dropna(how="all")

    # 6) (Tuỳ chọn) Resample
    if resample_rule:
        df = df.set_index("time")
        if resample_ohlc:
            ohlc = df[["open", "high", "low", "close"]].resample(resample_rule).ohlc()
            ohlc.columns = [c[0] for c in ohlc.columns]
            others = df.drop(columns=[c for c in ["open", "high", "low", "close"] if c in df.columns])
            agg_dict = {}
            for col in others.columns:
                if col.endswith("volume"):
                    agg_dict[col] = "sum"
                else:
                    agg_dict[col] = "last"
            if len(others.columns) > 0:
                others = others.resample(resample_rule).agg(agg_dict)
                df = pd.concat([ohlc, others], axis=1)
            else:
                df = ohlc
        else:
            df = df.resample(resample_rule).mean(numeric_only=True)

        if fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "bfill":
            df = df.bfill()
        elif fill_method != "none":
            raise ValueError(f"fill_method không hợp lệ: {fill_method}")

        df = df.reset_index()

    return df

# file: features.py
def make_features(
    df: pd.DataFrame,
    *,
    ma_fast: int = 10,
    ma_slow: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
    ret_lags: int = 3,            # số lượng lag return: ret_1, ret_2, ..., ret_k
    add_time_feats: bool = True,  # thêm đặc trưng theo lịch (giờ, thứ)
    add_candle_feats: bool = True,# thêm đặc trưng thân/nến
    drop_na: bool = True          # có drop các hàng bị NA sau khi tính rolling không
) -> tuple[pd.DataFrame, list[str]]:
    """
    Tạo features từ OHLCV. Trả về:
      - X: DataFrame chỉ chứa các cột feature, index ~ time của df
      - feature_names: danh sách tên feature theo đúng thứ tự (để export ONNX & dùng trong MT5)

    Yêu cầu df có các cột: time, open, high, low, close
    (tick_volume/spread/real_volume nếu có sẽ dùng thêm được).
    """

    # 0) Sao chép để không làm thay đổi df gốc
    data = df.copy()

    # 1) Tính return logarit một bước (an toàn với chia 0)
    data["ret_1"] = np.log(data["close"] / data["close"].shift(1))

    # 2) Tạo thêm các lag của return (ret_2 ... ret_k)
    for k in range(2, ret_lags + 1):
        data[f"ret_{k}"] = data["ret_1"].shift(k - 1)  # lag theo đúng mốc thời gian

    # 3) Đường trung bình động (MA) & slope
    data["ma_fast"] = data["close"].rolling(ma_fast).mean()
    data["ma_slow"] = data["close"].rolling(ma_slow).mean()
    # slope gần đúng: chênh lệch MA hiện tại so với MA của 1 nến trước
    data["ma_fast_slope"] = data["ma_fast"] - data["ma_fast"].shift(1)
    data["ma_slow_slope"] = data["ma_slow"] - data["ma_slow"].shift(1)
    # tín hiệu giao cắt
    data["ma_cross_up"] = (data["ma_fast"] > data["ma_slow"]).astype(int)

    # 4) ATR (Average True Range) & TrueRange
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    data["true_range"] = np.maximum(tr1, np.maximum(tr2, tr3))
    data["atr"] = data["true_range"].rolling(atr_period).mean()

    # 5) Rolling volatility (độ lệch chuẩn return)
    data["ret_std"] = data["ret_1"].rolling(atr_period).std()

    # 6) Candle structure (nếu bật)
    if add_candle_feats:
        body = (data["close"] - data["open"]).abs()
        range_ = (data["high"] - data["low"]).replace(0, np.nan)  # tránh chia 0
        upper_wick = (data["high"] - data[["open", "close"]].max(axis=1)).clip(lower=0)
        lower_wick = (data[["open", "close"]].min(axis=1) - data["low"]).clip(lower=0)

        data["c_body"] = body
        data["c_range"] = range_
        data["c_body_ratio"] = (body / range_).clip(upper=5)          # giới hạn để tránh outlier
        data["c_upper_wick_ratio"] = (upper_wick / range_).clip(upper=5)
        data["c_lower_wick_ratio"] = (lower_wick / range_).clip(upper=5)

    # 7) Time features (nếu bật) — không one-hot, để đơn giản khi port sang MT5
    if add_time_feats:
        # Nếu 'time' có tz, convert về UTC rồi lấy giờ/weekday; nếu không, dùng trực tiếp
        t = data["time"].dt.tz_convert("UTC") if data["time"].dt.tz is not None else data["time"]
        data["hour"] = t.dt.hour
        data["weekday"] = t.dt.weekday       # 0=Mon ... 6=Sun
        # chuẩn hoá về [0,1] để model dễ học hơn nhưng vẫn đơn giản để tái tạo trong MT5
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
        data["wk_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
        data["wk_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)

    # 8) Dùng tick_volume/spread nếu tồn tại
    if "tick_volume" in data.columns:
        data["vol_ema"] = data["tick_volume"].ewm(span=atr_period, adjust=False).mean()
    if "spread" in data.columns:
        data["spread_ema"] = data["spread"].ewm(span=atr_period, adjust=False).mean()

    # 9) Tập hợp danh sách feature cần dùng (giữ thứ tự ổn định để export ONNX/MT5)
    feature_names = []
    # returns
    feature_names += [f"ret_{k}" for k in range(1, ret_lags + 1)]
    # MA & slope & cross
    feature_names += ["ma_fast", "ma_slow", "ma_fast_slope", "ma_slow_slope", "ma_cross_up"]
    # ATR/vol
    feature_names += ["true_range", "atr", "ret_std"]
    # candle
    if add_candle_feats:
        feature_names += ["c_body", "c_range", "c_body_ratio", "c_upper_wick_ratio", "c_lower_wick_ratio"]
    # time
    if add_time_feats:
        feature_names += ["hour", "weekday", "hour_sin", "hour_cos", "wk_sin", "wk_cos"]
    """
    # volume/spread extras
    if "tick_volume" in data.columns:
        feature_names += ["vol_ema"]
    if "spread" in data.columns:
        feature_names += ["spread_ema"]
    """
    # 10) Tạo ma trận X từ các cột feature
    X = data[feature_names].copy()

    # 11) Xử lý NA phát sinh do rolling/shift
    if drop_na:
        # đồng bộ: khi X drop NA thì df gốc cũng nên canh theo index đó cho label ở bước sau
        mask = X.notna().all(axis=1)
        X = X[mask]
        # Trả về X; còn DataFrame gốc để dùng cho label sẽ cần được cắt cùng mask ở ngoài
    return X, feature_names

# Hàm 3: tạo nhãn lable
def make_labels_tp_sl(
    df,
    *,
    horizon=24,                 # số nến nhìn tới trước (ví dụ H1 -> 24h)
    take_profit=0.0020,         # delta giá để TP (EURUSD 0.0020 ~ 20 pips)
    stop_loss=0.0010,           # delta giá để SL (EURUSD 0.0010 ~ 10 pips)
    label_name="label",         # tên cột nhãn trả về
    prefer_tp=True ,             # nếu cùng nến vừa chạm TP vừa chạm SL: True -> TP thắng
    markup=0.0002
):
    """
    Tạo nhãn 0/1 theo luật TP/SL trong khung dự báo 'horizon'.
    - 1: Buy thắng (hit TP trước SL, hoặc không hit nhưng giá cuối >= entry)
    - 0: Buy thua (hit SL trước TP, hoặc không hit nhưng giá cuối < entry)
    Trả về: (y, info)
      - y: Series nhãn 0/1 (có thể có NaN ở cuối do không đủ horizon)
      - info: DataFrame phụ để phân tích (t_hit, outcome, tp_price, sl_price, close_future)
    """
    # đảm bảo có cột cần
    req = {"time", "high", "low", "close"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    n = len(df)
    y = np.full(n, np.nan, dtype=float)

    # pre-allocate debug info
    t_hit = np.full(n, np.nan)      # số nến mất để TP/SL
    outcome = np.array([""] * n, dtype=object)
    tp_price = np.full(n, np.nan)
    sl_price = np.full(n, np.nan)
    close_future = np.full(n, np.nan)

    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    for i in range(n):
        entry = closes[i]
        tp = entry + take_profit+markup
        sl = entry - (stop_loss-markup)
        tp_price[i] = tp
        sl_price[i] = sl

        # cửa sổ nhìn tới trước
        j_end = min(i + horizon, n - 1)
        if i == n - 1 or j_end <= i:
            # không còn dữ liệu phía trước
            continue

        # duyệt tuần tự để lấy sự kiện "xảy ra trước"
        hit = None        # ('TP' hoặc 'SL')
        steps = None      # số nến đến khi hit
        for j in range(i + 1, j_end + 1):
            # kiểm tra cùng nến: có thể high>=tp và low<=sl
            hit_tp = highs[j] >= tp
            hit_sl = lows[j]  <= sl
            if hit_tp and hit_sl:
                # cùng nến: ưu tiên theo prefer_tp
                hit = "TP" if prefer_tp else "SL"
                steps = j - i
                break
            elif hit_tp:
                hit = "TP"
                steps = j - i
                break
            elif hit_sl:
                hit = "SL"
                steps = j - i
                break

        if hit is not None:
            # có hit TP/SL trong horizon
            t_hit[i] = steps
            outcome[i] = hit
            y[i] = 1.0 if hit == "TP" else 0.0
        else:
            # không hit: so sánh close ở cuối cửa sổ
            cf = closes[j_end]
            close_future[i] = cf
            y[i] = 1.0 if cf >= entry else 0.0
            outcome[i] = "HOLD"
            t_hit[i] = j_end - i

    # đóng gói Series/DataFrame trả về
    y_series = pd.Series(y, index=df.index, name=label_name)
    info = pd.DataFrame({
        "entry": closes,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "t_hit": t_hit,
        "outcome": outcome,
        "close_future": close_future
    }, index=df.index)

    return y_series, info

# ====== TRAIN & EVALUATE (CatBoost) ======
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)

def _safe_auc(y_true, y_prob):
    """Tránh lỗi khi y_true chỉ có 1 lớp."""
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan

def _metrics(y_true, y_prob, threshold=0.5):
    """Tính metrics cơ bản từ xác suất."""
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "auc":  _safe_auc(y_true, y_prob),
        "logloss": log_loss(y_true, np.clip(y_prob, 1e-6, 1-1e-6)),
    }
    return m, y_pred
# ====== TIME SPLITS ======


def time_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    fw_ratio: float = 0.15,          # phần còn lại → FW; có thể bỏ qua và tự tính = 1 - train - valid
):
    """
    Chia theo THỨ TỰ THỜI GIAN (không shuffle):
    - train: đoạn đầu
    - valid: đoạn giữa (tuning/early-stop)
    - fw   : đoạn CUỐI (forward-test, out-of-sample)

    Trả về dict: {'train':(Xtr,ytr), 'valid':(...), 'fw':(...)}
    """
    assert len(X) == len(y), "X và y phải cùng số dòng"
    n = len(X)

    if fw_ratio is None:
        fw_ratio = 1.0 - train_ratio - valid_ratio
    assert 0 < train_ratio < 1 and 0 <= valid_ratio < 1 and 0 <= fw_ratio < 1, "tỉ lệ không hợp lệ"
    assert abs(train_ratio + valid_ratio + fw_ratio - 1.0) < 1e-6, "tổng tỉ lệ phải = 1"

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    # phần còn lại → fw
    n_fw = n - n_train - n_valid

    i0 = 0
    i1 = n_train
    i2 = n_train + n_valid
    i3 = n

    Xtr, ytr = X.iloc[i0:i1], y.iloc[i0:i1]
    Xva, yva = X.iloc[i1:i2], y.iloc[i1:i2]
    Xfw, yfw = X.iloc[i2:i3], y.iloc[i2:i3]

    return {
        "train": (Xtr, ytr),
        "valid": (Xva, yva),
        "fw":    (Xfw, yfw),
        "idx":   (i0, i1, i2, i3)   # để in/log nếu cần
    }

def train_evaluate_catboost(
    Xtr, ytr, Xva, yva, Xfw, yfw,
    *,
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=None,          # ví dụ: [w_for_class_0, w_for_class_1]
    random_seed=42,
    early_stopping_rounds=50,
    threshold=0.5                # ngưỡng phân loại khi tính metrics
):
    """
    Train CatBoost và đánh giá Train/Valid/FW.
    Trả về:
      model, metrics_dict, preds_dict
    """
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_seed=random_seed,
        class_weights=class_weights,
        verbose=False
    )

    # Fit với early stopping dựa trên VALID
    model.fit(
        Xtr, ytr,
        eval_set=(Xva, yva),
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds
    )

    # Xác suất class=1 cho 3 bộ
    p_tr = model.predict_proba(Xtr)[:, 1]
    p_va = model.predict_proba(Xva)[:, 1]
    p_fw = model.predict_proba(Xfw)[:, 1]

    # Tính metrics
    m_tr, yhat_tr = _metrics(ytr, p_tr, threshold)
    m_va, yhat_va = _metrics(yva, p_va, threshold)
    m_fw, yhat_fw = _metrics(yfw, p_fw, threshold)

    metrics = {"train": m_tr, "valid": m_va, "fw": m_fw}
    preds = {
        "train": {"y_prob": p_tr, "y_pred": yhat_tr},
        "valid": {"y_prob": p_va, "y_pred": yhat_va},
        "fw":    {"y_prob": p_fw, "y_pred": yhat_fw},
    }

    # In tóm tắt cho lớp xem
    print("\n=== CatBoost results (threshold=%.2f) ===" % threshold)
    for split, m in metrics.items():
        print(f"[{split.upper()}] ACC={m['acc']:.3f}  F1={m['f1']:.3f}  "
              f"PREC={m['prec']:.3f}  REC={m['rec']:.3f}  "
              f"AUC={m['auc']:.3f}  LOGLOSS={m['logloss']:.4f}")

    return model, metrics, preds

# ====== PLOT RESULTS ======


def plot_results(ytrue_dict, preds_dict, metrics_dict, *, threshold=0.5):
    """
    Vẽ 3 loại hình:
      - Confusion Matrix
      - ROC Curve
      - Phân bố xác suất dự đoán
    cho cả Train / Valid / Forward Test.
    
    ytrue_dict: {"train": ytr, "valid": yva, "fw": yfw}
    preds_dict: {"train": {"y_prob":..., "y_pred":...}, ...}
    metrics_dict: kết quả metrics từ train_evaluate_catboost
    """

    splits = ["train", "valid", "fw"]
    titles = {"train":"Train", "valid":"Validation", "fw":"Forward Test"}

    # === 1. Confusion Matrix ===
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for i, split in enumerate(splits):
        y_true = ytrue_dict[split]
        y_pred = preds_dict[split]["y_pred"]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    ax=axes[i], xticklabels=[0,1], yticklabels=[0,1])
        axes[i].set_title(f"Confusion Matrix - {titles[split]}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # === 2. ROC Curve ===
    fig, ax = plt.subplots(figsize=(6,6))
    for split in splits:
        y_true = ytrue_dict[split]
        y_prob = preds_dict[split]["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{titles[split]} AUC={roc_auc:.3f}")
    ax.plot([0,1], [0,1], "k--", label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.show()

    # === 3. Probability Distribution ===
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for i, split in enumerate(splits):
        y_prob = preds_dict[split]["y_prob"]
        sns.histplot(y_prob, bins=50, kde=True, ax=axes[i], color="blue")
        axes[i].axvline(threshold, color="red", linestyle="--", label="Threshold")
        axes[i].set_title(f"Prob. Distribution - {titles[split]}")
        axes[i].legend()
    plt.tight_layout()
    plt.show()

    # === 4. Print metrics in table form ===
    print("\n=== Metrics Summary ===")
    print(f"{'Split':<10} {'ACC':<6} {'F1':<6} {'PREC':<6} {'REC':<6} {'AUC':<6} {'LOGLOSS':<8}")
    for split, m in metrics_dict.items():
        print(f"{split:<10} {m['acc']:.3f}  {m['f1']:.3f}  {m['prec']:.3f}  "
              f"{m['rec']:.3f}  {m['auc']:.3f}  {m['logloss']:.4f}")
# hàm tester
def tester(model, X_train, y_train, X_valid, y_valid, X_fw, y_fw):
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              verbose=200,
              use_best_model=True)
    
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_valid = model.predict_proba(X_valid)[:, 1]
    y_pred_fw    = model.predict_proba(X_fw)[:, 1]

    ytrue_dict = {
        "train": y_train,
        "valid": y_valid,
        "fw":    y_fw,
    }
    preds = {
        "train": y_pred_train,
        "valid": y_pred_valid,
        "fw":    y_pred_fw,
    }
    return ytrue_dict, preds

import matplotlib.pyplot as plt
import numpy as np
#hàm vẽ đánh giá lợi nhuận
def plot_trading_results(df, y_true, y_pred, entry_col="close", tp=0.0020, sl=0.0020, threshold=0.3):
    """
    Vẽ Winrate và Equity Curve dựa trên tín hiệu model.
    - df: DataFrame chứa dữ liệu (có cột close/time).
    - y_true: nhãn thật (0/1).
    - y_pred: xác suất dự đoán của model.
    - entry_col: cột dùng làm entry price (mặc định close).
    - tp/sl: mức take profit / stop loss tính theo % (vd: 0.0020 = 20 pip).
    - threshold: ngưỡng để ra quyết định BUY.
    """
    signals = (y_pred >= threshold).astype(int)  # 1 = BUY, 0 = no trade
    entry_prices = df[entry_col].iloc[-len(signals):].values

    equity = [10000]  # vốn ban đầu
    wins, total = 0, 0

    for i in range(len(signals)):
        if signals[i] == 1:  # có trade
            total += 1
            # giả sử buy, kiểm tra nhãn thật
            if y_true.iloc[i] == 1:
                equity.append(equity[-1] * (1 + tp))  # thắng
                wins += 1
            else:
                equity.append(equity[-1] * (1 - sl))  # thua
        else:
            equity.append(equity[-1])  # không trade giữ nguyên

    winrate = wins / total if total > 0 else 0

    # Vẽ biểu đồ
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Winrate
    ax[0].bar(["Winrate"], [winrate*100], color="green")
    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("Winrate (%)")
    ax[0].set_title(f"Winrate = {winrate:.2%} | Trades = {total}")

    # Equity curve
    ax[1].plot(equity, label="Equity Curve", color="blue")
    ax[1].set_title("Equity Curve")
    ax[1].set_xlabel("Trade #")
    ax[1].set_ylabel("Balance")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# Khối hàm chính Run toàn code
if __name__ == "__main__":
    # Ví dụ: lấy từ 2015-01-01 đến hiện tại
    start = dt.datetime(2015, 1, 1)
    end   = dt.datetime.now()
    # ví dụ đọc file H1, giữ nguyên timeframe, chuẩn hoá tên cột nếu khác chuẩn
    df = load_clean_csv(
        path="EURUSD_H1_2015_to_now.csv",
        cols_map={"tick_volume": "tick_volume"},  # nếu file đã đúng chuẩn có thể bỏ
        resample_rule=None                         # None = không resample
    )
    df = df[df["time"] >= "2015-01-01"].reset_index(drop=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"])
    print(df.dtypes)
    print(df.isna().sum())
    
    print(df.head(), df.shape)
    

    X, feat_names = make_features(
        df,
        ma_fast=50,
        ma_slow=100,
        rsi_period=14,
        atr_period=14,
        ret_lags=3,
        add_time_feats=True,
        add_candle_feats=True,
        drop_na=True
    )

    print("X shape:", X.shape)
    print("Top features:", feat_names[:10], "...", len(feat_names), "features")

    # giả sử bạn đã chạy:
    # X, feat_names = make_features(df, ...)

    y, info = make_labels_tp_sl(
        df.loc[X.index],       # dùng đúng index của X để không lệch thời gian
        horizon=60,
        take_profit=0.0020,
        stop_loss=0.001,
        label_name="label",
        prefer_tp=True,
        markup=0.0002
    )

    print("y value counts:\n", y.value_counts(dropna=True))
    print("Số NaN ở cuối (do không đủ horizon):", y.isna().sum())
    # cắt bỏ các hàng NaN tương ứng để train:
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    df_aligned = df.loc[X.index].reset_index(drop=True)
    X_trainable = X[mask]
    y_trainable = y[mask]
    print("X_trainable:", X_trainable.shape, "| y_trainable:", y_trainable.shape)
    # nếu muốn xem nhanh vài dòng debug:
    print(info.loc[mask].head())
    

    # Sau khi có X_trainable, y_trainable ở bước trước:
    spl = time_splits(
        X_trainable, y_trainable,
        train_ratio=0.7, valid_ratio=0.15, fw_ratio=0.15
    )

    (Xtr, ytr) = spl["train"]
    (Xva, yva) = spl["valid"]
    (Xfw, yfw) = spl["fw"]
    (i0, i1, i2, i3) = spl["idx"]

    ytr = ytr.astype(int)
    yva = yva.astype(int)
    yfw = yfw.astype(int)

    model, metrics, preds = train_evaluate_catboost(
        Xtr, ytr, Xva, yva, Xfw, yfw,
        iterations=2000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        early_stopping_rounds=60,
        threshold=0.45
    )
    # 4) Plot kết quả phân loại
    ytrue_dict = {"train": ytr, "valid": yva, "fw": yfw}
    plot_results(ytrue_dict, preds, metrics, threshold=0.5)

    # 5) Cắt df_fw ĐÚNG theo split đã dùng ở trên
    df_train = df_aligned.iloc[i0:i1].copy()
    df_fw = df_aligned.iloc[i2:i3].copy()
    # 6) Vẽ Winrate/Equity cho TRAIN
    plot_trading_results(df_train, ytr, preds["train"]["y_prob"], threshold=0.5)
    # 6) Vẽ Winrate/Equity (dùng xác suất FW từ preds)
    plot_trading_results(df_fw, yfw, preds["fw"]["y_prob"], threshold=0.5)

    #7)n Saving mô hình sang Onx sau khi có train
    # === Save model CatBoost thành ONNX ===
    model.save_model(
        "catboost_model.onnx",       # file xuất
        format="onnx",               # định dạng
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "CatBoost Forex Model"
        }
    )

    # === Save danh sách feature names (thứ tự rất quan trọng) ===
    with open("feat_names.txt", "w", encoding="utf-8") as f:
        for name in feat_names:
            f.write(name + "\n")

    print("✅ Saved: catboost_model.onnx & feat_names.txt")



