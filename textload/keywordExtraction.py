import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from krwordrank.word import KRWordRank
from krwordrank.sentence import summarize_with_sentences
from keybert import KeyBERT
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import nltk

#nltk 다운로드
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

### 외부 도구 초기화 ###
# 한국 NLP 토큰 초기화
okt = Okt()
# KoBART 모델 및 토큰 초기화
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
#KeyBERT 한국어 지원 모델
kw_model = KeyBERT(model = 'distiluse-base-multilingual-cased')

# 불용어 설정
stopwords = set([
    "그리고", "그런데", "하지만", "그래서", "이는", "이러한", "또한", "이후", 
    "대한", "통해", "위한", "있다", "하는", "에서", "이다", "하는", "한편",
    "현재", "경우", "때문", "우리", "최근", "까지", "가지", "등", "즉", "또",
    ### 직접적으로 보고서 리포트 보면서 추가 설정 ##
])  

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text) # 특수문자 제거
    words = okt.nouns(text) # 명사 추출
    words = [word for word in words if word not in stopwords and len(word) > 1]
    return ' '.join(words)

# TF-IDF 키워드 추출
def extract_keywords_tfidf(text, top_n=10):
    text = preprocess_text(text)
    if not text.strip():  # 텍스트가 비었거나 전처리 후 내용이 없는 경우
        return []
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()

        sorted_indices = scores.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in sorted_indices]
        return keywords
    except ValueError:
        return []

# KeyBERT 기반 추출
def extract_keywords_keybert(text, top_n = 10):
    if not text or text.strip() == "":
        return []
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words = None, top_n = top_n)
    return [word for word, score in keywords]

# KR-WordRank 기반 추출
def extract_keywords_krwordrank(text, top_n=5):
    try:
        if not text or len(text.strip()) < 20:
            return []

        # === 전처리 적용 ===
        preprocessed_text = preprocess_text(text)
        if len(preprocessed_text.strip()) < 5:
            return []

        texts = [preprocessed_text]
        wordrank_extractor = KRWordRank(min_count=1, max_length=10, verbose=False)
        beta = 0.85
        max_iter = 10
        keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

        if not keywords or len(keywords) < 2:
            return []

        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_keywords[:top_n]]

    except Exception as e:
        print(f"KR-WordRank 키워드 추출 오류: {e}")
        return []


# KoBART 요약 함수
def summarize_text_kobart(text):
    if not text or text.strip() =="":
        return "요약할 내용이 없습니다."
    
    # 텍스트 길이 제한(토크나이저 max_len 고려)
    if  len(text) > 1000:
        text = text[:1000]

    input_ids = tokenizer.encode(text, return_tensors = 'pt', max_length = 1024, truncation = True)
    summary_ids = model.generate(
        input_ids,
        max_length = 100,
        min_length = 10,
        length_penalty = 2.0,
        num_beams = 4,
        early_stopping = True
    )
    
    output = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    return output.strip()
