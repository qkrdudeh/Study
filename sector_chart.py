"""
한국 주식시장 섹터별 일별 수익률 차트
- 실데이터: pykrx 또는 FinanceDataReader로 KRX 데이터 수집
- 폴백: 데모 데이터로 차트 구조 확인
"""

import sys
import warnings
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── 한국어 폰트 설정 ───────────────────────────────────────────────────────────

def setup_korean_font():
    nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    try:
        fm.fontManager.addfont(nanum_path)
        prop = fm.FontProperties(fname=nanum_path)
        font_name = prop.get_name()
        matplotlib.rcParams["font.family"] = font_name
        matplotlib.rcParams["axes.unicode_minus"] = False
        return font_name
    except Exception:
        matplotlib.rcParams["axes.unicode_minus"] = False
        return None


FONT_NAME = setup_korean_font()


# ── 섹터 데이터 수집 ──────────────────────────────────────────────────────────

KOSPI_SECTORS = {
    "1001": "종합(KOSPI)",
    "1002": "대형주",
    "1003": "중형주",
    "1004": "소형주",
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
    "1026": "제조업",
}

# 주요 섹터 업종코드 (종합·제조업·규모 지수 제외)
DISPLAY_SECTOR_IDS = [
    "1005", "1006", "1007", "1008", "1009", "1010",
    "1011", "1012", "1013", "1014", "1015", "1016",
    "1017", "1018", "1019", "1020", "1021", "1022",
    "1023", "1024", "1025",
]

# 섹터별 대략적 시가총액 비중 (treemap 크기용)
SECTOR_WEIGHT = {
    "1005": 3.2, "1006": 0.8, "1007": 0.6, "1008": 5.1, "1009": 4.8,
    "1010": 0.5, "1011": 3.9, "1012": 2.4, "1013": 28.5, "1014": 1.2,
    "1015": 7.6, "1016": 2.1, "1017": 1.8, "1018": 2.3, "1019": 1.4,
    "1020": 2.9, "1021": 8.7, "1022": 4.1, "1023": 2.8, "1024": 1.5,
    "1025": 5.8,
}


def fetch_pykrx_data(date_str: str) -> pd.DataFrame | None:
    """pykrx로 KOSPI 업종 지수 수익률 수집."""
    try:
        from pykrx import stock

        prev_date = (
            datetime.strptime(date_str, "%Y%m%d") - timedelta(days=1)
        ).strftime("%Y%m%d")

        rows = []
        for sid in DISPLAY_SECTOR_IDS:
            try:
                df = stock.get_index_ohlcv_by_date(prev_date, date_str, sid)
                if df is not None and len(df) >= 2:
                    ret = (df["종가"].iloc[-1] / df["종가"].iloc[-2] - 1) * 100
                    rows.append({"섹터ID": sid, "섹터명": KOSPI_SECTORS[sid], "수익률": round(ret, 2)})
            except Exception:
                pass

        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass
    return None


def fetch_fdr_data(date_str: str) -> pd.DataFrame | None:
    """FinanceDataReader로 KOSPI 업종 지수 수익률 수집."""
    try:
        import FinanceDataReader as fdr

        end_dt = datetime.strptime(date_str, "%Y%m%d")
        start_dt = end_dt - timedelta(days=7)

        rows = []
        for sid, name in KOSPI_SECTORS.items():
            if sid not in DISPLAY_SECTOR_IDS:
                continue
            try:
                df = fdr.DataReader(
                    f"KRX/INDEX/KOSPI/{sid}",
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"),
                )
                if df is not None and len(df) >= 2:
                    ret = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
                    rows.append({"섹터ID": sid, "섹터명": name, "수익률": round(ret, 2)})
            except Exception:
                pass

        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass
    return None


def get_demo_data(date_str: str) -> pd.DataFrame:
    """실데이터 수집 실패 시 사용하는 데모 데이터 (정규분포 랜덤)."""
    np.random.seed(int(date_str) % 9999)
    kospi_return = np.random.normal(0.3, 0.8)

    rows = []
    for sid in DISPLAY_SECTOR_IDS:
        ret = round(kospi_return + np.random.normal(0, 1.5), 2)
        rows.append({"섹터ID": sid, "섹터명": KOSPI_SECTORS[sid], "수익률": ret})

    df = pd.DataFrame(rows)
    df["수익률"] = df["수익률"].clip(-8, 8)
    return df


def get_sector_data(date_str: str) -> tuple[pd.DataFrame, bool]:
    """실데이터 → pykrx → FDR → 데모 순서로 시도."""
    df = fetch_pykrx_data(date_str)
    if df is not None and not df.empty:
        return df, True

    df = fetch_fdr_data(date_str)
    if df is not None and not df.empty:
        return df, True

    return get_demo_data(date_str), False


# ── 색상 유틸 ─────────────────────────────────────────────────────────────────

def return_to_color(ret: float) -> str:
    """수익률 → red/green 그라디언트 색상."""
    if ret > 0:
        intensity = min(ret / 5.0, 1.0)
        r = int(255 * (1 - intensity * 0.6))
        g = int(150 + 105 * intensity)
        b = int(150 * (1 - intensity))
        return f"#{r:02x}{g:02x}{b:02x}"
    elif ret < 0:
        intensity = min(abs(ret) / 5.0, 1.0)
        r = int(150 + 105 * intensity)
        g = int(150 * (1 - intensity))
        b = int(150 * (1 - intensity * 0.6))
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        return "#888888"


# ── 트리맵 레이아웃 ───────────────────────────────────────────────────────────

def squarify(weights: list[float], x: float, y: float, w: float, h: float) -> list[dict]:
    """Squarify 알고리즘으로 트리맵 직사각형 좌표 계산."""
    if not weights:
        return []

    total = sum(weights)
    rects = []
    weights = sorted(zip(weights, range(len(weights))), reverse=True)

    def layout(items, x, y, w, h):
        if not items:
            return
        if len(items) == 1:
            rects.append({"x": x, "y": y, "w": w, "h": h, "idx": items[0][1]})
            return

        if w >= h:
            # 가로 분할
            row, rest = _worst_ratio_split(items, w, h)
            row_w = sum(v for v, _ in row) / total * w
            yy = y
            for v, idx in row:
                rh = v / sum(vv for vv, _ in row) * h
                rects.append({"x": x, "y": yy, "w": row_w, "h": rh, "idx": idx})
                yy += rh
            layout(rest, x + row_w, y, w - row_w, h)
        else:
            # 세로 분할
            row, rest = _worst_ratio_split(items, h, w)
            row_h = sum(v for v, _ in row) / total * h
            xx = x
            for v, idx in row:
                rw = v / sum(vv for vv, _ in row) * w
                rects.append({"x": xx, "y": y, "w": rw, "h": row_h, "idx": idx})
                xx += rw
            layout(rest, x, y + row_h, w, h - row_h)

    def _worst_ratio_split(items, long_side, short_side):
        best_ratio = float("inf")
        best_n = 1
        for n in range(1, len(items) + 1):
            row = items[:n]
            row_sum = sum(v for v, _ in row) / total
            cell_long = row_sum * long_side
            worst = max(
                max(
                    cell_long / (v / total / row_sum * short_side),
                    (v / total / row_sum * short_side) / cell_long,
                )
                if row_sum > 0 else float("inf")
                for v, _ in row
            )
            if worst < best_ratio:
                best_ratio = worst
                best_n = n
            else:
                break
        return items[:best_n], items[best_n:]

    layout(weights, x, y, w, h)
    return rects


def _build_treemap(df: pd.DataFrame) -> list[dict]:
    """섹터 데이터프레임 → 트리맵 직사각형 리스트."""
    df = df.copy()
    df["weight"] = df["섹터ID"].map(SECTOR_WEIGHT).fillna(1.0)
    df = df.reset_index(drop=True)

    weights = df["weight"].tolist()
    rects = squarify(weights, 0, 0, 1, 1)

    result = []
    for r in rects:
        row = df.iloc[r["idx"]]
        result.append({
            "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"],
            "섹터명": row["섹터명"],
            "수익률": row["수익률"],
            "color": return_to_color(row["수익률"]),
        })
    return result


# ── 메인 차트 ────────────────────────────────────────────────────────────────

def draw_chart(df: pd.DataFrame, date_str: str, is_real: bool, output_path: str):
    df_sorted = df.sort_values("수익률", ascending=True).reset_index(drop=True)
    treemap_rects = _build_treemap(df)

    fig = plt.figure(figsize=(20, 13), facecolor="#0d1117")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, hspace=0.35)

    # ── 제목 ──────────────────────────────────────────────────────────────────
    date_display = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:]}"
    data_label = "" if is_real else "  [데모 데이터]"
    fig.text(
        0.5, 0.955,
        f"한국 주식시장 (KOSPI) 섹터별 일별 수익률  |  {date_display}{data_label}",
        ha="center", va="center",
        fontsize=22, fontweight="bold", color="#e6edf3",
    )

    # ── KOSPI 종합 수익률 요약 ────────────────────────────────────────────────
    avg_ret = df["수익률"].mean()
    pos_cnt = (df["수익률"] > 0).sum()
    neg_cnt = (df["수익률"] < 0).sum()
    summary = f"평균 수익률: {avg_ret:+.2f}%   상승 {pos_cnt}개   하락 {neg_cnt}개"
    fig.text(
        0.5, 0.922,
        summary,
        ha="center", va="center",
        fontsize=13, color="#8b949e",
    )

    gs_top = fig.add_gridspec(1, 1, left=0.02, right=0.98, top=0.905, bottom=0.52)
    gs_bot = fig.add_gridspec(1, 2, left=0.02, right=0.98, top=0.48, bottom=0.05,
                               wspace=0.05)

    # ── 1. 트리맵 ─────────────────────────────────────────────────────────────
    ax_tree = fig.add_subplot(gs_top[0])
    ax_tree.set_facecolor("#161b22")
    ax_tree.set_xlim(0, 1)
    ax_tree.set_ylim(0, 1)
    ax_tree.axis("off")
    ax_tree.set_title("시가총액 비중 트리맵", color="#8b949e", fontsize=12, pad=8)

    for r in treemap_rects:
        pad = 0.003
        rect = mpatches.FancyBboxPatch(
            (r["x"] + pad, r["y"] + pad),
            r["w"] - 2 * pad, r["h"] - 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor=r["color"],
            edgecolor="#0d1117",
            linewidth=1.5,
        )
        ax_tree.add_patch(rect)

        cx = r["x"] + r["w"] / 2
        cy = r["y"] + r["h"] / 2
        area = r["w"] * r["h"]

        if area > 0.008:
            fontsize_name = max(7, min(14, area * 200))
            fontsize_ret = max(6, min(12, area * 170))
            text_color = "white" if abs(r["수익률"]) > 1 else "#cccccc"
            ax_tree.text(cx, cy + 0.018, r["섹터명"],
                        ha="center", va="center",
                        fontsize=fontsize_name, fontweight="bold",
                        color=text_color)
            sign = "▲" if r["수익률"] > 0 else ("▼" if r["수익률"] < 0 else "")
            ax_tree.text(cx, cy - 0.022, f"{sign}{abs(r['수익률']):.2f}%",
                        ha="center", va="center",
                        fontsize=fontsize_ret, color=text_color)
        elif area > 0.002:
            fontsize_ret = max(5, min(9, area * 250))
            ax_tree.text(cx, cy, f"{r['수익률']:+.1f}%",
                        ha="center", va="center",
                        fontsize=fontsize_ret, color="white")

    # ── 2. 수평 막대 차트 ────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs_bot[0])
    ax_bar.set_facecolor("#161b22")
    fig.patch.set_facecolor("#0d1117")

    colors = [return_to_color(r) for r in df_sorted["수익률"]]
    bars = ax_bar.barh(
        df_sorted["섹터명"],
        df_sorted["수익률"],
        color=colors,
        height=0.72,
        edgecolor="#0d1117",
        linewidth=0.5,
    )

    ax_bar.axvline(0, color="#30363d", linewidth=1.2, linestyle="-")
    ax_bar.set_xlabel("수익률 (%)", color="#8b949e", fontsize=11)
    ax_bar.set_title("섹터별 수익률 (낮은 순)", color="#8b949e", fontsize=12, pad=10)
    ax_bar.tick_params(colors="#8b949e", labelsize=10)
    ax_bar.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
    ax_bar.set_facecolor("#161b22")
    ax_bar.xaxis.label.set_color("#8b949e")

    for bar, ret in zip(bars, df_sorted["수익률"]):
        x = bar.get_width()
        offset = 0.05 if x >= 0 else -0.05
        ha = "left" if x >= 0 else "right"
        sign = "▲" if ret > 0 else ("▼" if ret < 0 else "")
        ax_bar.text(
            x + offset, bar.get_y() + bar.get_height() / 2,
            f"{sign}{abs(ret):.2f}%",
            va="center", ha=ha,
            fontsize=9, color="#e6edf3",
        )

    # ── 3. 버블 차트 ────────────────────────────────────────────────────────
    ax_bub = fig.add_subplot(gs_bot[1])
    ax_bub.set_facecolor("#161b22")

    df_plot = df.copy()
    df_plot["weight"] = df_plot["섹터ID"].map(SECTOR_WEIGHT).fillna(1.0)
    df_plot["color"] = df_plot["수익률"].apply(return_to_color)
    df_plot["rank"] = df_plot["수익률"].rank()

    scatter = ax_bub.scatter(
        df_plot["rank"],
        df_plot["수익률"],
        s=df_plot["weight"] * 60,
        c=df_plot["color"],
        alpha=0.85,
        edgecolors="#30363d",
        linewidths=0.8,
    )

    for _, row in df_plot.iterrows():
        ax_bub.annotate(
            row["섹터명"],
            (row["rank"], row["수익률"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=7.5, color="#c9d1d9",
            rotation=30,
        )

    ax_bub.axhline(0, color="#30363d", linewidth=1, linestyle="--")
    ax_bub.set_xlabel("수익률 순위", color="#8b949e", fontsize=11)
    ax_bub.set_ylabel("수익률 (%)", color="#8b949e", fontsize=11)
    ax_bub.set_title("섹터별 수익률 버블 차트 (원 크기 = 시총 비중)", color="#8b949e",
                     fontsize=12, pad=10)
    ax_bub.tick_params(colors="#8b949e", labelsize=9)
    ax_bub.spines[["top", "right", "left", "bottom"]].set_color("#30363d")
    ax_bub.xaxis.label.set_color("#8b949e")
    ax_bub.yaxis.label.set_color("#8b949e")

    # ── 컬러바 범례 ───────────────────────────────────────────────────────────
    cmap_vals = np.linspace(-5, 5, 256)
    cmap_colors = [return_to_color(v) for v in cmap_vals]
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    cmap = ListedColormap(cmap_colors)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(-5, 5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_bar, ax_bub], orientation="horizontal",
                        fraction=0.015, pad=0.18, aspect=40)
    cbar.set_label("수익률 (%)", color="#8b949e", fontsize=10)
    cbar.ax.tick_params(colors="#8b949e", labelsize=9)
    cbar.outline.set_edgecolor("#30363d")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"차트 저장 완료: {output_path}")


# ── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    # 날짜 인자 처리: python sector_chart.py YYYYMMDD
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        # 가장 최근 평일
        today = datetime.today()
        offset = max(1, (today.weekday() >= 5) * (today.weekday() - 4))
        date_str = (today - timedelta(days=offset)).strftime("%Y%m%d")

    print(f"기준일: {date_str}")
    df, is_real = get_sector_data(date_str)

    data_source = "실시간 KRX 데이터" if is_real else "데모 데이터 (네트워크 미연결)"
    print(f"데이터 소스: {data_source}")
    print(df.to_string(index=False))

    output_path = f"sector_return_{date_str}.png"
    draw_chart(df, date_str, is_real, output_path)


if __name__ == "__main__":
    main()
