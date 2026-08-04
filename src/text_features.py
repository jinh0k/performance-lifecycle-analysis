"""시놉시스에서 작품 주제를 키워드로 뽑아 라벨링하는 코드.

LDA로 장르별 주제를 살펴본 뒤, 자주 나오는 21개 주제(사랑, 전쟁, 가족, 마법, 음악,
천재, 꿈 …)를 골랐다. 여기서는 그 주제 키워드가 각 작품 줄거리에 들어 있는지를 0/1로
표시해서, 작품의 소재를 예측 모델(classify.py)이 쓸 수 있는 특성으로 바꾼다.

방식이 단순 키워드 매칭이라 몇 가지 한계가 있다:
* 문맥이나 부정 표현("가난하지 않은")은 못 잡는다.
* 아래 DUPLICATED_KEYWORDS처럼 한 키워드가 여러 주제에 걸치면 컬럼끼리 상관이
  생긴다. 트리 모델 예측에는 큰 문제 없지만, 중요도 해석할 때는 감안해야 한다.
* 'Men'의 '그'는 한 글자라 엉뚱하게 걸리는 경우가 있다(개선 여지).
"""
from __future__ import annotations

import pandas as pd

# 주제 -> 키워드 목록.
KEYWORDS: dict[str, list[str]] = {
    "Magic": ["마법", "마술"],
    "Friendship": ["친구", "우정"],
    "Family": ["가족", "엄마", "아빠", "동생"],
    "Music": ["음악", "기교", "노래"],
    "Genius": ["천재"],
    "Craftsmanship": ["기술", "장인", "혁명", "작곡가", "화가", "예술"],
    "Life": ["인생", "삶"],
    "Love": ["사랑", "결혼", "연애", "아름다운"],
    "War": ["전쟁", "싸움", "결투", "군인"],
    "Dreams": ["꿈"],
    "Human Relationships": ["관계", "연애", "친구"],
    "Emotions": ["감정", "마음", "울음", "감동", "아름다운"],
    "Conflict": ["갈등", "충돌"],
    "Social Problems": ["사회문제", "빈부격차", "현실", "돈", "가난한"],
    "Human Nature": ["인간", "본성"],
    "Memories": ["기억", "추억"],
    "Her": ["그녀", "여자", "아내"],
    "Humor": ["유머", "농담", "웃음"],
    "Challenge": ["도전", "시련", "모험", "여행", "시작"],
    "Men": ["남자", "남편", "그"],
    "Novel": ["명작", "소설", "원작"],
}

# 여러 주제에 겹쳐 다중공선성을 유발하는 키워드(참고용).
DUPLICATED_KEYWORDS = {
    "친구": ["Friendship", "Human Relationships"],
    "연애": ["Love", "Human Relationships"],
    "아름다운": ["Love", "Emotions"],
}


def label_synopsis(text: object) -> dict[str, int]:
    """하나의 시놉시스 텍스트에 대해 주제별 0/1 라벨 dict를 반환한다."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return {
        topic: int(any(kw in text for kw in kws))
        for topic, kws in KEYWORDS.items()
    }


def add_topic_labels(df: pd.DataFrame, synopsis_col: str = "시놉시스") -> pd.DataFrame:
    """DataFrame의 시놉시스 컬럼에서 주제 라벨 컬럼들을 만들어 붙인다."""
    labels = df[synopsis_col].apply(label_synopsis).apply(pd.Series)
    return pd.concat([df, labels], axis=1)


if __name__ == "__main__":
    demo = pd.DataFrame(
        {"시놉시스": ["마법과 우정을 다룬 이야기", "전쟁 속 가족의 사랑", None]}
    )
    print(add_topic_labels(demo).to_string())
