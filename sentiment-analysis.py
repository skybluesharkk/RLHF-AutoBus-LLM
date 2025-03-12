from transformers import pipeline

# 감성 분석 모델 로드 (다국어 지원)
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def get_reward_from_feedback(feedback):
    """자연어 피드백에서 Reward(보상) 분석"""
    sentiment = sentiment_pipeline(feedback)
    score = sentiment[0]["label"]

    # 보상 점수 변환 (예: 별점 5점 기준)
    if "5 stars" in score:
        return 1  # 매우 긍정적
    elif "4 stars" in score:
        return 0.5  # 긍정적
    elif "3 stars" in score:
        return 0  # 중립적
    elif "2 stars" in score:
        return -0.5  # 부정적
    elif "1 star" in score:
        return -1  # 매우 부정적
    else:
        return 0  # 기본값 (예외 처리)

# 테스트 예제
print(get_reward_from_feedback("좋아!"))  # 보상: 1
print(get_reward_from_feedback("그렇게 하면 안 돼"))  # 보상: -1
print(get_reward_from_feedback("괜찮네"))  # 보상: 0
