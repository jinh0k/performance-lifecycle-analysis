"""KOPIS 예매 데이터를 분석용으로 다듬는 전처리 코드.

KOPIS raw 예매 로그(xlsx 여러 개)를 받아서, 분석 대상만 추리고 필요한 파생변수를
만든 뒤 CSV로 저장한다.

대상은 문화소비가 활발한 서울·경기·경상의 뮤지컬·연극으로 좁혔고, 코로나 여파가
남은 2022년 8월 이전 공연은 분포가 왜곡될 수 있어 뺐다. 공연기간(종료일-시작일),
수상실적 개수, 좌석등급 같은 파생변수를 붙이고, 0원 티켓이나 연령·성별·출연진이
비어 있는 행처럼 신뢰하기 어려운 데이터는 정리했다. 오픈런 여부는 open_run 인자로
받는다(기본 'Y').

입력: KOPIS raw xlsx (여러 파일). 출력: 정제된 CSV.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REGIONS = ["서울", "경기도", "경상도"]
GENRES = ["뮤지컬", "연극"]
MIN_PERIOD_DAYS = 14
COVID_CUTOFF = pd.Timestamp("2022-08-01")  # 코로나 특수 구간 제외

COLS_TO_DROP = [
    "전송사업자코드", "전송사업자명", "공연시설코드", "개관연도", "주소",
    "무대시설_오케스트라피트 여부", "무대시설_연습실 여부", "무대시설_분장실 여부",
    "무대시설_무대넓이", "입장권고유번호", "예매/취소방식코드", "예매/취소방식명(전송처)",
    "결제수단코드", "결제수단명(전송처)", "할인금액", "할인종류코드",
    "할인종류명(관리시스템)", "할인종류명(전송처)", "세부장르명", "제작진내용",
    "기획제작사명", "원작자명", "극작가명", "판매시작일시", "판매종료일시",
    "단독판매여부", "판매좌석수", "예매/취소금액", "수상실적", "좌석등급_dict", "좌석등급",
]


def count_awards(awards: object) -> int:
    """수상실적 문자열의 항목 개수(쉼표 구분). NaN이면 0."""
    if pd.isna(awards):
        return 0
    return len(str(awards).split(","))


def parse_grades(grade_str: str) -> dict[str, int]:
    """'좌석등급' 문자열을 {등급명: 금액} 사전으로 파싱한다."""
    grades = re.findall(r"(\D+)\((\d+)\)", str(grade_str))
    return {g.strip().lstrip(","): int(amount) for g, amount in grades}


def find_closest_grade(amount: float, grade_dict: dict[str, int]) -> str:
    """장당금액에 가장 가까운 좌석 등급명을 찾는다."""
    if not grade_dict:
        return "Not grade"
    return min(grade_dict, key=lambda k: abs(grade_dict[k] - amount))


def preprocess(df: pd.DataFrame, open_run: str = "Y") -> pd.DataFrame:
    """단일 raw DataFrame을 정제한다.

    open_run: '오픈런 여부'로 남길 값. 원본이 'Y'를 남겼으므로 기본 'Y'.
    """
    df = df.copy()

    # (수정 1) 지역 AND 장르 로 필터 — 원본의 OR 버그 교정.
    df = df[df["공연지역명"].isin(REGIONS) & df["장르명"].isin(GENRES)].copy()

    df["장애인석"] = df["장애인석"].fillna(0)
    df = df[df["소요시간"].notna()].copy()
    df["수상실적_개수"] = df["수상실적"].apply(count_awards)
    df["좌석등급"] = df["좌석등급"].fillna("X")
    df = df[df["장당금액"] != 0].copy()

    # (수정 2) 오픈런 필터를 인자로 명시.
    df = df[df["오픈런 여부"] == open_run].copy()

    df = df[(df["연령"] != 0) & (df["성별"] != 0)].copy()   # 남:1, 여:2
    df = df[df["출연진내용"].notna()].copy()

    for col in ["공연일시", "공연시작일자", "공연종료일자"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df[df["공연시작일자"] >= COVID_CUTOFF].copy()

    df["공연기간"] = df["공연종료일자"] - df["공연시작일자"]
    df = df[df["공연기간"] > pd.Timedelta(days=MIN_PERIOD_DAYS)].copy()

    df["좌석등급_dict"] = df["좌석등급"].apply(parse_grades)
    df["좌석등급_부여"] = df.apply(
        lambda r: find_closest_grade(r["장당금액"], r["좌석등급_dict"]), axis=1
    )

    df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)
    return df


def process_folder(data_dir: str | Path, save_dir: str | Path, open_run: str = "Y") -> None:
    """폴더 내 모든 xlsx를 전처리해 CSV로 저장한다(원본의 [48:50] 슬라이스 제거)."""
    data_dir, save_dir = Path(data_dir), Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(sorted(data_dir.glob("*.xlsx"))):
        out = preprocess(pd.read_excel(path), open_run=open_run)
        out_path = save_dir / f"공연_23_{i}_{open_run}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"processed {path.name} -> {out_path.name} ({len(out)} rows)")


if __name__ == "__main__":
    print(__doc__)
    print("사용: process_folder(raw_xlsx_dir, save_dir)")
