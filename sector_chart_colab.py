"""
KOSPI 섹터별 일별 등락률 히트맵 (Colab용)
run(end_date) 호출로 Plotly 차트를 출력합니다.
"""

import warnings, numpy as np, pandas as pd, holidays as hl
from datetime import datetime, timedelta
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

SECTORS = {
    '1005':'음식료품','1006':'섬유의복','1007':'종이목재','1008':'화학',
    '1009':'의약품','1010':'비금속광물','1011':'철강금속','1012':'기계',
    '1013':'전기전자','1014':'의료정밀','1015':'운수장비','1016':'유통업',
    '1017':'전기가스업','1018':'건설업','1019':'운수창고업','1020':'통신업',
    '1021':'금융업','1022':'은행','1023':'증권','1024':'보험','1025':'서비스업',
}


def _resolve_date(end_date: str) -> str:
    if end_date:
        return end_date
    today = datetime.today()
    if today.weekday() >= 5:
        today -= timedelta(days=today.weekday() - 4)
    return today.strftime('%Y%m%d')


def _fetch_real(start_str: str, end_date: str):
    from pykrx import stock
    matrix = {}
    for sid, name in SECTORS.items():
        df = stock.get_index_ohlcv_by_date(start_str, end_date, sid)
        if df is not None and not df.empty:
            matrix[name] = df['종가'].pct_change() * 100
    if not matrix:
        return None
    result = pd.DataFrame(matrix).T
    result.columns = [d.strftime('%Y-%m-%d') for d in result.columns]
    return result.iloc[:, 1:].round(2)


def _make_demo(start_str: str, end_dt: datetime):
    s = datetime.strptime(start_str, '%Y%m%d')
    kr_hols = hl.country_holidays('KR', years=[s.year, end_dt.year])
    days = [s + timedelta(days=i) for i in range((end_dt - s).days + 1)
            if (s + timedelta(days=i)).weekday() < 5
            and (s + timedelta(days=i)) not in kr_hols]
    cols = [d.strftime('%Y-%m-%d') for d in days]
    np.random.seed(42)
    return pd.DataFrame(
        {n: np.clip((np.random.normal(.05, .7, len(cols)) +
                     np.random.normal(0, 1.1, len(cols))).round(2), -8, 8)
         for n in SECTORS.values()}, index=cols).T


def _get_data(end_date: str):
    end_dt    = datetime.strptime(end_date, '%Y%m%d')
    start_str = (end_dt - timedelta(days=40)).strftime('%Y%m%d')
    try:
        df = _fetch_real(start_str, end_date)
        if df is None or df.empty:
            raise ValueError()
        return df.iloc[:, -25:], True
    except Exception:
        return _make_demo(start_str, end_dt), False


def run(end_date: str = ''):
    end_date = _resolve_date(end_date)
    df, is_real = _get_data(end_date)

    src = '실시간 KRX' if is_real else '데모'
    print(f"{src} | {df.shape[0]}섹터 × {df.shape[1]}영업일 | 기준일 {end_date}")

    z       = df.values.tolist()
    dates   = [d[5:].replace('-', '/') for d in df.columns]   # MM/DD
    sectors = list(df.index)

    text = []
    for row in df.itertuples(index=False):
        text.append([
            f"{'▲' if v > 0 else '▼' if v < 0 else ' '}{abs(v):.2f}%"
            if not np.isnan(v) else '' for v in row
        ])

    hover = []
    for si, sector in enumerate(sectors):
        hover.append([
            f"<b>{sector}</b><br>{df.columns[di]}<br>"
            f"{'+' if z[si][di] > 0 else ''}{z[si][di]:.2f}%"
            for di in range(len(dates))
        ])

    colorscale = [
        [0.0,  '#c0392b'],
        [0.35, '#e88080'],
        [0.5,  '#f5f5f5'],
        [0.65, '#80c880'],
        [1.0,  '#196f3d'],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=dates, y=sectors,
        text=text, texttemplate='%{text}',
        hovertext=hover, hovertemplate='%{hovertext}<extra></extra>',
        colorscale=colorscale, zmid=0, zmin=-5, zmax=5,
        showscale=True,
        colorbar=dict(title='등락률(%)', tickvals=[-5,-3,-1,0,1,3,5],
                      thickness=14, len=0.8),
        xgap=2, ygap=2,
    ))

    note = '' if is_real else ' [데모]'
    end_disp = f"{end_date[:4]}.{end_date[4:6]}.{end_date[6:]}"
    fig.update_layout(
        title=dict(text=f'KOSPI 섹터별 일별 등락률 — 최근 1개월{note}  ({end_disp})',
                   font=dict(size=16), x=0.5),
        height=680,
        margin=dict(l=10, r=60, t=60, b=40),
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3', size=11),
        xaxis=dict(type='category', side='top', tickangle=-45,
                   tickfont=dict(size=11), gridcolor='#30363d', fixedrange=False),
        yaxis=dict(autorange='reversed', gridcolor='#30363d', fixedrange=False),
    )
    fig.show()
