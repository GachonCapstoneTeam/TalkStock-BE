from sentence_transformers  import SentenceTransformer, util
import pandas as pd
import numpy as np
from typing import List, Dict


# 사전 학습된 한국어 문장 임베딩 모델
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def keyword_relevance_score(doc: str, keywords: List[str]) -> float:
    """
    주어진 문서와 키워드 리스트의 유사도를 계산한다.
    문서와 키워드들의 임베딩 평균 간 cosine similarity를 반환.
    """
    if not doc or not keywords:
        return 0.0

    # 문서 임베딩
    doc_embedding = model.encode(doc, convert_to_tensor=True)
    # 키워드 임베딩 ( 리스트 내 단어들 각각)
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)
    # 키워드 임베딩 평균
    keyword_mean_embedding = keyword_embeddings.mean(dim=0)
    # 코사인 유사도 계산
    similarity = util.cos_sim(doc_embedding, keyword_mean_embedding).item()
    
    print("문서 임베딩:", doc_embedding)
    print("키워드 임베딩:", keyword_embeddings)

    return round(similarity, 4)

def evaluate_all_keywords(
    documents: List[str],
    all_keywords: Dict[str, List[List[str]]]
) -> pd.DataFrame:
    """
    여러 키워드 추출 방법에 대해 모든 문서의 관련성 점수를 계산한다.
    
    Parameters:
    - documents: 원문 리스트
    - all_keywords: {"TF-IDF": [[kw1], [kw2], ...], "KeyBERT": [...], ...}
    
    Returns:
    - DataFrame with relevance scores for each method
    """
    results = {'Document': documents}
    
    for method_name, keyword_lists in all_keywords.items():
        print(f"[INFO] Evaluating {method_name} ...")
        scores = [
            keyword_relevance_score(doc, keywords)
            for doc, keywords in zip(documents, keyword_lists)
        ]
        results[f'{method_name}_Score'] = scores

    return pd.DataFrame(results)