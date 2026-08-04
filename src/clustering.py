"""공연을 "판매 곡선의 모양"으로 묶는 군집화 코드.

공연마다 개막~종연 기간을 5개 구간으로 쪼개고, 구간별 판매량의 평균·표준편차·기울기를
뽑는다. 판매가 얼마나 빨리 퍼지고(기울기), 얼마나 크고(평균), 얼마나 들쭉날쭉한지(표준
편차)를 숫자로 옮긴 셈이다. 이걸 K-Means로 묶으면 Peak / Early Peak / Retention 같은
수명주기 유형이 갈린다. 군집 수는 실루엣 계수를 보고 정한다.

입력 데이터 스키마
------------------
이 모듈은 공연별 **일자별 판매 로그**가 필요하다(레포의 집계 CSV에는 없다).
long-format DataFrame을 기대한다:

    공연코드 | 판매일자(datetime) | 판매량(int)

원본 raw 예매 로그(전처리.ipynb 산출물)에서 공연코드+판매일자로 groupby 하면
만들 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_SEGMENTS = 5          # 각 공연 기간을 몇 구간으로 나눌지 (원본과 동일)
N_CLUSTERS = 3          # 최종 군집 수 (select_k로 근거 확인 가능)


def _segment_index(dates: pd.Series, n_segments: int) -> np.ndarray:
    """공연별 판매일자를 0..n_segments-1 구간 인덱스로 변환한다."""
    d = pd.to_datetime(dates)
    span = (d.max() - d.min()).days
    if span <= 0:
        return np.zeros(len(d), dtype=int)
    frac = (d - d.min()).dt.days / span         # 0.0 ~ 1.0
    idx = (frac * n_segments).astype(int).clip(0, n_segments - 1)
    return idx.to_numpy()


def _slope(y: np.ndarray) -> float:
    """1차 다항 회귀 기울기. 점이 2개 미만이면 0."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def build_segment_features(log: pd.DataFrame, n_segments: int = N_SEGMENTS) -> pd.DataFrame:
    """일자별 판매 로그 -> 공연코드별 구간 mean/std/slope 특징 테이블.

    log: ['공연코드', '판매일자', '판매량'] long-format.
    반환: index=공연코드, 컬럼= 구간별 mean/std/slope (n_segments * 3개).
    """
    rows: dict[str, dict[str, float]] = {}
    for code, g in log.groupby("공연코드"):
        seg = _segment_index(g["판매일자"], n_segments)
        feats: dict[str, float] = {}
        for s in range(n_segments):
            vals = g.loc[seg == s, "판매량"].to_numpy(dtype=float)
            feats[f"seg{s}_mean"] = vals.mean() if len(vals) else np.nan
            feats[f"seg{s}_std"] = vals.std() if len(vals) else np.nan
            feats[f"seg{s}_slope"] = _slope(vals)
        rows[code] = feats

    features = pd.DataFrame.from_dict(rows, orient="index")
    # 결측(관객 없던 구간)은 해당 컬럼 최소값으로 대체 (원본 로직 유지, 1줄로 정리).
    return features.fillna(features.min())


def select_k(X: np.ndarray, k_range: range = range(2, 8)) -> dict[int, float]:
    """실루엣 계수로 군집 수 후보를 평가해 {k: score} 반환(정량 근거)."""
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    return scores


def cluster_performances(
    features: pd.DataFrame, n_clusters: int = N_CLUSTERS
) -> pd.Series:
    """구간 특징을 표준화한 뒤 KMeans로 군집 라벨을 한 번만 계산한다."""
    X = StandardScaler().fit_transform(features.to_numpy())
    labels = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE
    ).fit_predict(X)
    return pd.Series(labels, index=features.index, name="Cluster")


def run(log: pd.DataFrame) -> pd.DataFrame:
    """전체 파이프라인: 로그 -> 특징 -> 군집 라벨이 붙은 특징 테이블."""
    features = build_segment_features(log)
    sil = select_k(StandardScaler().fit_transform(features.to_numpy()))
    print("실루엣 계수(군집 수 선택 근거):", {k: round(v, 3) for k, v in sil.items()})
    features = features.copy()
    features["Cluster"] = cluster_performances(features)
    return features


if __name__ == "__main__":
    print(__doc__)
    print("이 모듈은 공연별 일자별 판매 로그를 입력으로 요구합니다. "
          "run(log) 형태로 호출하세요.")
