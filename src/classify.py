"""공연 수명 유형(Cluster)을 예측하는 분류 모델.

clustering.py에서 나눈 수명 유형을 정답으로 놓고, 공연의 시설·장르·가격·관객 구성·
주제 라벨·지역 지표만으로 그 유형을 맞힐 수 있는지 보는 코드다.

범주형은 OrdinalEncoder로 숫자화하고 수치형은 그대로 뒀다(트리 모델이라 스케일링은
굳이 안 했다). 유형별 표본이 불균형해서 SMOTE로 늘리는데, 이걸 파이프라인 안에 넣어
교차검증 폴드마다 학습 데이터에만 적용되게 했다 — 안 그러면 성능이 부풀려진다.
의사결정나무와 랜덤포레스트를 5-fold 교차검증과 별도 테스트셋으로 비교하고,
불균형 데이터라 F1(macro)로 본다.

실행:  python src/classify.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "training_data_with_external.csv"
TARGET = "Cluster"
ID_COL = "공연코드"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """학습 데이터 CSV를 로드한다."""
    return pd.read_csv(path)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """식별자/타깃을 분리하고 X, y를 만든다.

    '장애인석'은 'N'(문자)과 정수가 섞여 있어 숫자로 강제 변환하고
    변환 실패값(예: 'N')은 0으로 채운다.
    """
    df = df.copy()
    if "장애인석" in df.columns:
        df["장애인석"] = pd.to_numeric(df["장애인석"], errors="coerce").fillna(0)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """범주형은 OrdinalEncoder, 수치형은 그대로 통과시키는 전처리기.

    트리 기반 모델을 쓰므로 스케일링은 하지 않는다. 인코더는 파이프라인
    내부에 있어 각 CV 폴드의 학습 부분에서만 fit 된다(누수 방지).
    """
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical),
            ("num", "passthrough", numeric),
        ]
    )


def make_pipeline(preprocessor: ColumnTransformer, classifier) -> ImbPipeline:
    """전처리 → SMOTE → 분류기로 이어지는 누수 없는 파이프라인.

    class 2가 매우 희소(약 10개)하므로 SMOTE k_neighbors를 작게 잡아
    폴드 안에서 최소 샘플 부족으로 에러 나는 것을 방지한다.
    """
    return ImbPipeline(
        steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("clf", classifier),
        ]
    )


def evaluate(X: pd.DataFrame, y: pd.Series) -> None:
    """DecisionTree와 RandomForest를 CV + 홀드아웃으로 비교 평가한다."""
    preprocessor = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    for name, clf in models.items():
        pipe = make_pipeline(preprocessor, clf)

        # SMOTE가 파이프라인 안에 있으므로 폴드별로만 오버샘플링됨(누수 없음).
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\n===== {name} =====")
        print(f"CV F1(macro) : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test F1(macro): {test_f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))


def main() -> None:
    df = load_data()
    print(f"데이터: {df.shape[0]}행 x {df.shape[1]}열")
    print("Cluster 분포:\n", df[TARGET].value_counts().sort_index())
    X, y = build_xy(df)
    evaluate(X, y)


if __name__ == "__main__":
    main()
