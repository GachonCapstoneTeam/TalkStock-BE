import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt

okt = Okt()

def preprocess(text):
    nouns = okt.nouns(text)
    return [noun for noun in nouns if len(noun) > 1]

def evaluate_keywords(all_keywords: dict, documents: list):
    tokenized_texts = [preprocess(doc) for doc in documents]
    dictionary = Dictionary(tokenized_texts)

    # Coherence
    coherence_scores = {
        method: CoherenceModel(topics=keywords, texts=tokenized_texts, dictionary=dictionary, coherence='c_v').get_coherence()
        for method, keywords in all_keywords.items()
    }

    # Diversity
    def diversity_score(keyword_lists):
        all_keywords = [" ".join(kw_list) for kw_list in keyword_lists]
        vecs = TfidfVectorizer().fit_transform(all_keywords)
        sim_matrix = cosine_similarity(vecs)
        avg_sim = np.mean(sim_matrix)
        return 1 - avg_sim

    diversity_scores = {
        method: diversity_score(keywords)
        for method, keywords in all_keywords.items()
    }

    # Representativeness
    def representativeness_score(keywords_list, documents):
        summaries = [" ".join(kw) for kw in keywords_list]
        vec = TfidfVectorizer().fit(documents + summaries)
        doc_vec = vec.transform(documents)
        summary_vec = vec.transform(summaries)
        sim = cosine_similarity(doc_vec, summary_vec)
        return np.mean(sim.diagonal())

    represent_scores = {
        method: representativeness_score(keywords, documents)
        for method, keywords in all_keywords.items()
    }

    # 통합 결과 리턴
    return {
        "Coherence": coherence_scores,
        "Diversity": diversity_scores,
        "Representativeness": represent_scores
    }