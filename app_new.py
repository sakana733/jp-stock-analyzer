
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import re
import html

st.set_page_config(page_title="日本株分析・拡張版", layout="wide")
st.title("日本株分析アプリ・機能拡張版")
st.caption("株価・テクニカル分析・財務情報・信用取引情報・バックテスト")

with st.sidebar:
    st.header("分析設定")
    code = st.text_input("証券コード（4桁）", "7203").strip()
    period = st.selectbox("取得期間", ["1y", "2y", "5y"], index=1)
    cost_bps = st.number_input(
        "片道取引コスト（bp）", min_value=0, max_value=200,
        value=10, step=5
    )
    api_key = st.text_input(
        "J-Quants APIキー（任意）", type="password"
    )
    run = st.button("分析する", type="primary")

def get_prices(symbol, period):
    df = yf.download(
        symbol, period=period, auto_adjust=True,
        progress=False, multi_level_index=False
    )
    if df.empty:
        raise ValueError("株価データを取得できませんでした。")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "Close"]).copy()
    if len(df) < 100:
        raise ValueError("分析に必要な株価データが不足しています。")
    return df

def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    df["MA25"] = close.rolling(25).mean()
    df["MA75"] = close.rolling(75).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI"] = 50

    df["BuySignal"] = (
        (close > df["MA25"]) &
        (df["MA25"] > df["MA75"]) &
        (df["MACD"] > df["Signal"]) &
        df["RSI"].between(40, 70)
    )
    return df

def safe_number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan

def get_yahoo_japan_ratios(symbol):
    """Yahoo!ファイナンス日本版の掲載値を取得。取得失敗時は欠損値を返す。"""
    result = {"ROE": np.nan, "自己資本比率": np.nan}
    try:
        url = f"https://finance.yahoo.co.jp/quote/{symbol}"
        response = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15
        )
        response.raise_for_status()
        # ページ内の埋め込みJSONは引用符がエスケープされている場合がある。
        page = html.unescape(response.text)
        for key, label in (("roe", "ROE"), ("equityRatio", "自己資本比率")):
            # 対象のキーに続くオブジェクトからvalueを取得し、別の項目との誤対応を防ぐ。
            pattern = (
                r'["\\]*' + re.escape(key) +
                r'["\\]*\s*:\s*\{.{0,400}?["\\]*value["\\]*\s*:\s*'
                r'["\\]*(-?\d+(?:\.\d+)?)["\\]*'
            )
            match = re.search(pattern, page, flags=re.DOTALL)
            if match:
                result[label] = float(match.group(1)) / 100.0
    except (requests.RequestException, ValueError):
        pass
    return result


def get_company_name(symbol):
    """銘柄名を取得。取得できない場合は証券コードを表示する。"""
    # 日本語の銘柄名が取得できる場合はYahoo!ファイナンス日本版を優先。
    try:
        response = requests.get(
            f"https://finance.yahoo.co.jp/quote/{symbol}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        )
        response.raise_for_status()
        match = re.search(
            r'<meta\s+[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)',
            response.text, re.IGNORECASE
        )
        if not match:
            match = re.search(
                r'<meta\s+[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:title["\']',
                response.text, re.IGNORECASE
            )
        if match:
            title = html.unescape(match.group(1)).strip()
            # サイト名や証券コードなどの付記を取り除く。
            name = re.split(r'（\d{4}[A-Z]?）|【| - |｜|\|', title, maxsplit=1)[0].strip()
            if name and name != symbol:
                return name
    except (requests.RequestException, ValueError):
        pass
    try:
        info = yf.Ticker(symbol).info
        name = info.get("longName") or info.get("shortName")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return symbol


def get_financials(symbol):
   
    ticker = yf.Ticker(symbol)
    info = ticker.info
    st.write("財務データの取得状況：", len(info), "項目")
    jp_ratios = get_yahoo_japan_ratios(symbol)
    return {
        "時価総額": safe_number(info.get("marketCap")),
        "PER": safe_number(info.get("trailingPE")),
        "PBR": safe_number(info.get("priceToBook")),
        "ROE": (jp_ratios["ROE"] if np.isfinite(jp_ratios["ROE"])
                else safe_number(info.get("returnOnEquity"))),
        "営業利益率": safe_number(info.get("operatingMargins")),
        "自己資本比率": jp_ratios["自己資本比率"],
        "売上高": safe_number(info.get("totalRevenue")),
        "純利益": safe_number(info.get("netIncomeToCommon")),
    }

def get_margin(code, api_key):
    if not api_key:
        return pd.DataFrame(), "J-Quants APIキーが未入力です。"
    try:
        url = "https://api.jquants.com/v2/markets/margin-interest"
        response = requests.get(
            url,
            headers={"x-api-key": api_key},
            params={"code": code},
            timeout=20
        )
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data", [])
        if not records:
            return pd.DataFrame(), "信用取引データがありません。"
        return pd.DataFrame(records), ""
    except requests.RequestException as e:
        return pd.DataFrame(), f"信用取引データの取得に失敗: {e}"

def technical_score(row):
    checks = [
        row["Close"] > row["MA25"],
        row["MA25"] > row["MA75"],
        row["MACD"] > row["Signal"],
        40 <= row["RSI"] <= 70,
    ]
    return sum(bool(x) for x in checks) / 4 * 10

def financial_score(fin):
    checks = []
    if np.isfinite(fin["ROE"]):
        checks.append(fin["ROE"] >= 0.08)
    if np.isfinite(fin["営業利益率"]):
        checks.append(fin["営業利益率"] >= 0.08)
    if np.isfinite(fin["自己資本比率"]):
        checks.append(fin["自己資本比率"] >= 0.40)
    if np.isfinite(fin["純利益"]):
        checks.append(fin["純利益"] > 0)
    if not checks:
        return np.nan
    return sum(checks) / len(checks) * 10

def margin_score(margin):
    if margin.empty:
        return np.nan, np.nan
    df = margin.copy()
    date_col = "PubDate" if "PubDate" in df.columns else "Date"
    if date_col not in df.columns:
        return np.nan, np.nan
    df = df.sort_values(date_col)
    last = df.iloc[-1]
    long_vol = safe_number(last.get("LongVol"))
    short_vol = safe_number(last.get("ShrtVol"))
    if not np.isfinite(long_vol) or not np.isfinite(short_vol):
        return np.nan, np.nan
    if short_vol <= 0:
        return np.nan, np.nan
    ratio = long_vol / short_vol
    score = 7 if ratio < 1 else 6 if ratio < 3 else 5 if ratio < 5 else 4
    return score, ratio

def composite_score(tech, fin, margin):
    parts = [
        (tech, 0.4),
        (fin, 0.4),
        (margin, 0.2),
    ]
    available = [
        (score, weight)
        for score, weight in parts if np.isfinite(score)
    ]
    if not available:
        return np.nan, 0
    weight_sum = sum(weight for _, weight in available)
    score = sum(score * weight for score, weight in available) / weight_sum
    return score, weight_sum

def backtest(df, holding_days, cost_bps):
    """
    当日の終値でシグナルを確認し、翌営業日の始値で購入。
    holding_days営業日後の終値で売却。
    重複ポジションを持たない簡易バックテスト。
    """
    trades = []
    i = 75
    while i + holding_days < len(df):
        if bool(df["BuySignal"].iloc[i]):
            entry_i = i + 1
            exit_i = i + holding_days
            buy = float(df["Open"].iloc[entry_i])
            sell = float(df["Close"].iloc[exit_i])
            if buy > 0:
                gross = sell / buy - 1
                net = gross - 2 * cost_bps / 10000
                trades.append({
                    "シグナル日": str(df.index[i].date()),
                    "購入日": str(df.index[entry_i].date()),
                    "売却日": str(df.index[exit_i].date()),
                    "購入価格": buy,
                    "売却価格": sell,
                    "損益率": net,
                })
            i = exit_i + 1
        else:
            i += 1
    return pd.DataFrame(trades)

if run:
   
    code = code.upper()

    if not (len(code) == 4 and code.isalnum() and code[:3].isdigit()):
        st.error("証券コードは7203や285Aの形式で入力してください。")
        st.stop()

    symbol = code + ".T"

    try:
        with st.spinner("株価を取得しています..."):
            df = add_indicators(get_prices(symbol, period))
    except Exception as e:
        st.error(f"株価データを取得できませんでした: {e}")
        st.stop()

    latest = df.iloc[-1]
    st.success(f"{symbol} の株価分析が完了しました。")

    st.subheader("1. 株価とテクニカル指標")
    company_name = get_company_name(symbol)
    st.markdown(f"### {company_name}（{code}）")
    c1, c2, c3 = st.columns(3)
    # 前営業日の終値を基準に前日比を算出（休場日は直近の取引日を使用）
    current_close = float(df["Close"].iloc[-1])
    previous_close = float(df["Close"].iloc[-2])
    price_change = current_close - previous_close
    price_change_pct = price_change / previous_close * 100 if previous_close != 0 else np.nan
    if np.isfinite(price_change_pct):
        delta_text = f"{price_change:+,.1f} 円 ({price_change_pct:+.2f}%)"
    else:
        delta_text = f"{price_change:+,.1f} 円"
    c1.metric("終値", f"{current_close:,.1f} 円", delta=delta_text)
    c2.metric("RSI", f"{latest['RSI']:.1f}")
    c3.metric("テクニカル評価", f"{technical_score(latest):.1f} / 10")

    fig = go.Figure()
    for column, label in [
        ("Close", "株価"), ("MA25", "25日移動平均"),
        ("MA75", "75日移動平均")
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[column], name=label
        ))
    fig.update_layout(height=450, xaxis_title="日付", yaxis_title="円")
    st.plotly_chart(fig, use_container_width=True)

    # MACDとシグナル線のグラフ
    st.subheader("MACD（12・26・9）")
    macd_fig = go.Figure()

    macd_fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MACD"],
        name="MACD"
    ))

    macd_fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Signal"],
        name="シグナル線"
    ))

    macd_fig.add_trace(go.Bar(
        x=df.index,
        y=df["MACD"] - df["Signal"],
        name="ヒストグラム"
    ))

    macd_fig.add_hline(y=0, line_dash="dash")
    macd_fig.update_layout(
        height=300,
        xaxis_title="日付",
        yaxis_title="MACD"
    )

    st.plotly_chart(macd_fig, use_container_width=True)

    st.subheader("2. 財務情報")
    fin = {}
    try:
        fin = get_financials(symbol)
        rows = []
        for name, value in fin.items():
            if not np.isfinite(value):
                display = "取得できません"
            elif name in ["ROE", "営業利益率", "自己資本比率"]:
                display = f"{value * 100:.2f}%"
            elif name in ["PER", "PBR"]:
                display = f"{value:.2f} 倍"
            else:
                display = f"{value:,.0f}"
            rows.append({"項目": name, "値": display})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("出典：Yahoo!ファイナンス日本版（ROE・自己資本比率の取得成功時）、Yahoo Finance / yfinance（その他）。取得時点の情報です。Yahoo日本版から取得できない場合、ROEはyfinanceの値、自己資本比率は未取得となります。")
    except Exception as e:
        st.warning(f"財務情報を取得できませんでした: {e}")

    st.subheader("3. 信用取引情報")
    margin, message = get_margin(code, api_key)
    if message:
        st.info(message)
    if not margin.empty:
        st.dataframe(margin.tail(20), use_container_width=True)
        st.download_button(
            "信用取引データをCSV保存",
            margin.to_csv(index=False).encode("utf-8-sig"),
            "margin.csv", "text/csv"
        )
    st.caption(
        "出典：J-Quants API。信用取引データの利用には"
        "対応する契約プランが必要です。"
    )

    st.subheader("4. 総合スコア")
    tech = technical_score(latest)
    fs = financial_score(fin) if fin else np.nan
    ms, credit_ratio = margin_score(margin)
    total, coverage = composite_score(tech, fs, ms)

    a, b, c, d = st.columns(4)
    a.metric("テクニカル", f"{tech:.1f}/10")
    b.metric("財務", f"{fs:.1f}/10" if np.isfinite(fs) else "未取得")
    c.metric("信用取引", f"{ms:.1f}/10" if np.isfinite(ms) else "未取得")
    d.metric("総合評価", f"{total:.1f}/10" if np.isfinite(total) else "算出不可")

    st.write(f"評価に利用できた項目の重み：{coverage * 100:.0f}%")
    if np.isfinite(credit_ratio):
        st.write(f"信用倍率（取得値から計算）：{credit_ratio:.2f} 倍")
    st.caption(
        "試験的な評価ルールです。重みはテクニカル40%、財務40%、"
        "信用取引20%。欠損項目は除外して再計算しています。"
    )

    st.subheader("5. 将来リターンのバックテスト")
    st.write(
        "過去に買いシグナルが出た場合、翌営業日の始値で購入し、"
        "一定営業日後の終値で売却したと仮定します。"
    )

    summaries = []
    for days in [5, 20, 60]:
        trades = backtest(df, days, cost_bps)
        if trades.empty:
            summaries.append({
                "保有日数": days, "取引回数": 0,
                "平均損益率": np.nan, "勝率": np.nan
            })
            continue

        summaries.append({
            "保有日数": days,
            "取引回数": len(trades),
            "平均損益率": trades["損益率"].mean() * 100,
            "勝率": (trades["損益率"] > 0).mean() * 100,
        })
        with st.expander(f"{days}営業日保有の取引履歴"):
            st.dataframe(trades, use_container_width=True)
            st.download_button(
                f"{days}営業日の結果をCSV保存",
                trades.to_csv(index=False).encode("utf-8-sig"),
                f"backtest_{days}d.csv", "text/csv"
            )

    result = pd.DataFrame(summaries)
    st.dataframe(
        result.style.format({
            "平均損益率": "{:.2f}%",
            "勝率": "{:.1f}%"
        }, na_rep="—"),
        use_container_width=True
    )

    st.warning(
        "バックテストにはテクニカル指標のみを使用しています。"
        "現在の財務・信用取引データを過去の売買判断に使うと、"
        "未来の情報を利用する問題が起きるため除外しています。"
        "実際の約定価格、流動性、税金などは再現していません。"
    )

    st.subheader("6. 株価データのダウンロード")
    st.download_button(
        "分析データをCSV保存",
        df.to_csv().encode("utf-8-sig"),
        f"{code}_analysis.csv",
        "text/csv"
    )

else:
    st.info("左側に証券コードを入力して「分析する」を押してください。")
