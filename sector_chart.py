"""
한국 주식시장 KOSPI 섹터별 일별 수익률 히트맵
- 행: 섹터
- 열: 최근 1개월 거래일
- 셀: 당일 등락률(%)
"""

import sys
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── 한국어 폰트 ───────────────────────────────────────────────────────────────

def setup_korean_font():
    path = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
    try:
        fm.fontManager.addfont(path)
        prop = fm.FontProperties(fname=path)
        name = prop.get_name()
        matplotlib.rcParams["font.family"] = name
        matplotlib.rcParams["axes.unicode_minus"] = False
        return name
    except Exception:
        matplotlib.rcParams["axes.unicode_minus"] = False
        return None


setup_korean_font()


# ── 섹터 정의 ─────────────────────────────────────────────────────────────────

SECTORS = {
    "1005": "음식료품",
    "1006": "섬유의복",
    "1007": "종이목재",
    "1008": "화학",
    "1009": "의약품",
    "1010": "비금속광물",
    "1011": "철강금속",
    "1012": "기계",
    "1013": "전기전자",
    "1014": "의료정밀",
    "1015": "운수장비",
    "1016": "유통업",
    "1017": "전기가스업",
    "1018": "건설업",
    "1019": "운수창고업",
    "1020": "통신업",
    "1021": "금융업",
    "1022": "은행",
    "1023": "증권",
    "1024": "보험",
    "1025": "서비스업",
}


# ── 데이터 수집 ───────────────────────────────────────────────────────────────

def fetch_pykrx(start: str, end: str) -> pd.DataFrame | None:
    """pykrx로 섹터별 일별 등락률 수집 → DataFrame[섹터명 × 날짜]."""
    try:
        from pykrx import stock

        matrix = {}
        for sid, name in SECTORS.items():
            df = stock.get_index_ohlcv_by_date(start, end, sid)
            if df is None or df.empty:
                continue
            # 전일 대비 등락률
            ret = df["종가"].pct_change() * 100
            matrix[name] = ret

        if not matrix:
            return None

        result = pd.DataFrame(matrix).T  # 섹터 × 날짜
        result.columns = [d.strftime("%Y-%m-%d") for d in result.columns]
        result = result.iloc[:, 1:]      # 첫 날은 NaN이므로 제거
        return result.round(2)
    except Exception:
        return None


def make_demo(start: str, end: str) -> pd.DataFrame:
    """거래일 생성 후 정규분포 랜덤 등락률로 데모 데이터 구성."""
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    trading_days = [
        s + timedelta(days=i)
        for i in range((e - s).days + 1)
        if (s + timedelta(days=i)).weekday() < 5  # 월~금
    ]
    date_cols = [d.strftime("%Y-%m-%d") for d in trading_days]

    np.random.seed(42)
    rows = {}
    for name in SECTORS.values():
        # 시장 공통 팩터 + 섹터 고유 팩터
        market = np.random.normal(0.05, 0.7, len(date_cols))
        sector = np.random.normal(0.0, 1.1, len(date_cols))
        rets = (market + sector).round(2)
        rets = np.clip(rets, -8, 8)
        rows[name] = rets

    return pd.DataFrame(rows, index=date_cols).T


def get_data(end_date: str) -> tuple[pd.DataFrame, bool]:
    """end_date 기준 최근 1개월 데이터 수집."""
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=40)   # 영업일 기준 약 1개월 확보
    start_str = start_dt.strftime("%Y%m%d")

    df = fetch_pykrx(start_str, end_date)
    if df is not None and not df.empty:
        # 최근 25거래일만 사용
        return df.iloc[:, -25:], True

    return make_demo(start_str, end_date), False


# ── 색상 ─────────────────────────────────────────────────────────────────────

def cell_color(val: float) -> tuple:
    """등락률 → RGBA 배경색."""
    if np.isnan(val):
        return (0.15, 0.15, 0.15, 1.0)
    cap = 4.0
    t = np.clip(val / cap, -1, 1)
    if t > 0:
        # 흰색 → 진한 초록
        r = 1 - t * 0.75
        g = 1 - t * 0.15
        b = 1 - t * 0.75
    elif t < 0:
        # 흰색 → 진한 빨강
        r = 1 + t * 0.15
        g = 1 + t * 0.85
        b = 1 + t * 0.85
    else:
        return (0.97, 0.97, 0.97, 1.0)
    return (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1), 1.0)


def text_color(bg: tuple) -> str:
    """배경색 밝기에 따라 글자색 결정."""
    r, g, b, _ = bg
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "#222222"


# ── 차트 ──────────────────────────────────────────────────────────────────────

def draw(df: pd.DataFrame, end_date: str, is_real: bool, out: str):
    n_sectors, n_days = df.shape
    cell_w, cell_h = 2.1, 0.62

    fig_w = n_days * cell_w + 3.5
    fig_h = n_sectors * cell_h + 2.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, n_days)
    ax.set_ylim(0, n_sectors)
    ax.axis("off")

    # ── 셀 그리기 ──────────────────────────────────────────────────────────────
    for r_idx, sector in enumerate(df.index):
        y = n_sectors - 1 - r_idx          # 위에서 아래
        for c_idx, date in enumerate(df.columns):
            val = df.at[sector, date]
            bg = cell_color(val)
            tc = text_color(bg)

            ax.add_patch(plt.Rectangle(
                (c_idx + 0.02, y + 0.04),
                0.96, 0.92,
                facecolor=bg, edgecolor="#0d1117", linewidth=0.5,
                transform=ax.transData, clip_on=False,
            ))

            if not np.isnan(val):
                sign = "▲" if val > 0 else ("▼" if val < 0 else " ")
                label = f"{sign}{abs(val):.2f}%"
                ax.text(
                    c_idx + 0.5, y + 0.5, label,
                    ha="center", va="center",
                    fontsize=8.2, color=tc, fontweight="bold",
                )

    # ── 섹터명 (왼쪽) ─────────────────────────────────────────────────────────
    for r_idx, sector in enumerate(df.index):
        y = n_sectors - 1 - r_idx
        ax.text(
            -0.15, y + 0.5, sector,
            ha="right", va="center",
            fontsize=10, color="#e6edf3",
        )

    # ── 날짜 (상단) ───────────────────────────────────────────────────────────
    for c_idx, date in enumerate(df.columns):
        mmdd = date[5:]   # MM-DD
        ax.text(
            c_idx + 0.5, n_sectors + 0.15, mmdd,
            ha="center", va="bottom",
            fontsize=8.5, color="#8b949e", rotation=45,
        )

    # ── 구분선 (5일 단위) ─────────────────────────────────────────────────────
    for c in range(0, n_days + 1, 5):
        ax.axvline(c, color="#30363d", linewidth=0.8, ymin=0, ymax=1)

    # ── 제목 ──────────────────────────────────────────────────────────────────
    end_disp = f"{end_date[:4]}.{end_date[4:6]}.{end_date[6:]}"
    data_note = "" if is_real else "  [데모 데이터]"
    fig.text(
        0.5, 0.98,
        f"KOSPI 섹터별 일별 등락률  |  최근 1개월  (기준일 {end_disp}){data_note}",
        ha="center", va="top",
        fontsize=15, fontweight="bold", color="#e6edf3",
    )

    # ── 범례 ──────────────────────────────────────────────────────────────────
    for pct, label in [(-4, "-4%↓"), (-2, "-2%"), (0, "0%"),
                       (2, "+2%"), (4, "+4%↑")]:
        bg = cell_color(float(pct))
        fig.text(
            0.5 + pct * 0.018, 0.012,
            label,
            ha="center", va="bottom",
            fontsize=8.5,
            color=text_color(bg),
            bbox=dict(facecolor=bg, edgecolor="none", boxstyle="round,pad=0.3"),
        )

    plt.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.05)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"저장 완료: {out}")


# ── 실행 ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        end_date = sys.argv[1]
    else:
        today = datetime.today()
        if today.weekday() >= 5:
            today -= timedelta(days=today.weekday() - 4)
        end_date = today.strftime("%Y%m%d")

    print(f"기준일: {end_date}")
    df, is_real = get_data(end_date)
    print(f"데이터: {'실시간 KRX' if is_real else '데모'} | "
          f"섹터 {df.shape[0]}개 × {df.shape[1]}일")
    print(df.to_string())

    out = f"sector_heatmap_{end_date}.png"
    draw(df, end_date, is_real, out)


if __name__ == "__main__":
    main()
