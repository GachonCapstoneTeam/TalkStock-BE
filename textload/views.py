import os
import django
import logging
import pandas as pd
from collections import defaultdict
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import nltk
import re
import openai
from krwordrank.word import KRWordRank
from django.http import JsonResponse
from nltk.corpus import stopwords  # stopwords import
from . import pdfToText as pdf #pdf파일을 크롤링
from . import crawling as crawling #네이버리포트를 크롤링

# Download stopwords punkt(if not already done)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Django 환경설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE","TalkStockReport.settings")
django.setup()

# 로깅 설정
logging.getLogger("urllib3").setLevel(logging.WARNING)

### 외부 도구 초기화 ###
# 한국 NLP 토큰 초기화
okt = Okt()
# KoBART 모델 및 토큰 초기화
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
#KeyBERT 한국어 지원 모델
kw_model = KeyBERT(model = 'distiluse-base-multilingual-cased')
# 발급받은 API 키 설정
client = openai.Client(api_key = "OPENAI_API_KEY")

# Define Korean stopwords
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
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()

    sorted_indices = scores.argsort()[-top_n:][::-1]
    keywords = [feature_names[i] for i in sorted_indices]
    return keywords

# KeyBERT 기반 추출
def extract_keywords_keybert(text, top_n = 10):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words = None, top_n = top_n)
    return [word for word, score in keywords]

# KR-WordRank 기반 추출
def extract_keywords_krwordrank(text,top_n=5):
    try:
        if not text or len(text.strip())<20:
            return []
        
        texts = [text] # 리스트 형태로 입력
        wordrank_extractor = KRWordRank(min_count =1, max_length= 10, verbose=False)
        beta = 0.85     # PageRank의 decaying factor
        max_iter = 10   
        keywords, rank, graph = wordrank_extractor.extract(texts,beta, max_iter)
        
        if not keywords or len(keywords) < 2:
            return []
        sorted_keywords = sorted(keywords.items(), key = lambda x: x[1], reverse=True) 
        return [word for word, score in sorted_keywords[:top_n]]
    
    except Exception as e:
        print(f"KR-WordRank 키워드 추출 오류: {e}")
        return []

# TF-IDF + KR-WordRank 결합 키워드 추출
def extract_combined_keywords(text, top_n=10):
    try:
        tfidf_keywords = extract_keywords_tfidf(text, top_n * 2)
        krwordrank_keywords = extract_keywords_krwordrank(text, top_n * 2)
        tfidf_set = set(tfidf_keywords)

        # 교집합 우선 반환
        combined = [word for word in krwordrank_keywords if word in tfidf_set]
        if combined:
            return combined[:top_n]
        else:
            # 겹치는게 없으면 TF-IDF 결과 반환
            return tfidf_keywords[:top_n]
    
    except Exception as e:
        print(f"키워드 추출 오류: {e}")
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

# PDF 다운로드 및 텍스트 추출 함수
#pdf_content = pdf.download_and_process_pdf2(pdf_url,company)


# MongoDB 연결 함수
def connect_to_mongo():
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)  # SSL 비활성화
        
        db = client['report_database']
        
        print("Successfully connected to MongoDB!")
        return db

    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def insert_data_into_mongo(data):
    db = connect_to_mongo()
    collection = db["reports"]

    # 데이터 삽입
    if isinstance(data, list):
        collection.insert_many(data)
    else:
        collection.insert_one(data)

    print("데이터가 MongoDB에 성공적으로 저장되었습니다!!")

# Global DataFrame to store all reports
df = pd.DataFrame()

# 네이버 리포트 크롤링
def fetch_data():
    # 네이버 리포트 크롤링할 페이지 수 지정
    all_reports = crawling.fetch_all_reports(pages=5)

    # DataFrame에 데이터 추가
    for report in all_reports:
        pdf_url = report.get('PDF URL')
        company = report.get('증권사')
        if pdf_url != "PDF 없음" and company:
            report["PDF Content"] = pdf.download_and_process_pdf2(pdf_url, company)
        else:
            report["PDF Content"] = ""
    return all_reports

# # OpenAI API를 사용한 요약 함수
# def summarize_text_with_gpt(text):

#     if not text or text.strip() == "":
#         return "No content available for summarization."

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "당신은 한국어 금융 보고서를 1문장으로 요약하는 AI야. 절대 영어를 사용하지 마. 모든 요약을 반드시 한국어로 작성해"},
#                 {"role": "user", "content": f"다음 금융 보고서를 한국어로 1문장으로 요약해줘. 절대 영어를 사용하지 마:\n{text}"}
#             ],
#             max_tokens=100
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error: {e}"    

# all report를 가져오기
fetch_data()

df['Content'] = df['Content'].apply(lambda x: x.replace("\n", "") if pd.notnull(x) else x)
df['PDF Content'] = df['PDF Content'].apply(lambda x: x.replace("\n", "") if pd.notnull(x) else x)
df['keywords'] = df['Content'].astype(str).apply(lambda x: extract_combined_keywords(x, top_n = 5))
df['TFIDF'] = df['Content'].astype(str).apply(lambda x: extract_keywords_tfidf(x, top_n = 5))
df['KRWordRank'] = df['PDF Content'].astype(str).apply(lambda x: extract_keywords_krwordrank(x, top_n = 5))
df['keybert'] = df['Content'].astype(str).apply(lambda x: extract_keywords_keybert(x, top_n = 5))
df['Summary'] = df['Content'].astype(str).apply(lambda x: summarize_text_kobart(x))


df_cleaned = df.dropna(subset=['PDF Content'])

# 업종별 키워드 모음 생성
industry_kewords_map = defaultdict(list)

for idx, row in df.iterrows():
    industry = row['업종']
    keywords = row['KRWordRank']

    if industry != "Unknown" and isinstance(keywords, str):
        keyword_list = eval(keywords)
        industry_kewords_map[industry].extend(keyword_list)

# 업종별 키워드 중복 제거 및 정리
for industry, keywords in industry_kewords_map.items():
    unique_keywords = list(set(keywords))
    industry_kewords_map[industry] = unique_keywords

# Unknown 업종 -> 키워드 기반 업종 매칭
def match_industry_by_keywords(keywords):
    if not isinstance(keywords, str):
        return "Unknown"
    
    keyword_list = eval(keywords)
    match_scores = {}

    for industry, industry_keywords in industry_kewords_map.items():
        matches = set(keyword_list) & set(industry_keywords)
        match_scores[industry] = len(matches)

    # 가장 많이 겹친 업종 변환(없으면 Unknown)
    matched_industry = max(match_scores, key=match_scores.get)
    if match_scores[matched_industry] == 0:
        return "Unknown"
    else:
        return matched_industry
    
df['업종'] = df.apply(lambda row: match_industry_by_keywords(row['KRWordRank']) if row['업종'] == "Unknown" else row['업종'], axis = 1)

# Save the updated DataFrame to a new Excel fileç
output_file_path = "all_reports_with_summary_keywords.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Summarization complete! The updated file is saved as: {output_file_path}")

# 딕셔너리 형태로 변환
data_to_save = df.to_dict('records')

# Save to MongoDB
insert_data_into_mongo(data_to_save)
print("데이터가 MongoDB에 저장되었습니다!!")

@api_view()
def hello_world(request):
    return Response({"originaltext" : "하늘이 장차 그 사람에게 큰 사명을 주려 할 때는 반드시 먼저 그의 마음과 뜻을 흔들어 고통스럽게 하고,  힘줄과 뼈를 굶주리게 하여 궁핍하게 만들어 그가 하고자 하는 일을 흔들고 어지럽게 하나니그것은 타고난 작고 못난 성품을 인내로써 담금질을 하여 하늘의 사명을 능히 감당할 만 하도록 그 기국과 역량을 키워주기 위함이다."})


@api_view(['GET'])
def content(request):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['report_database']
    collection = db['reports']

    # 작성일 기준으로 정렬하여 최신 데이터 반환
    data_cursor = collection.find({}, {"_id": 0}).sort("작성일", -1)  # 작성일 기준 내림차순 정렬
    data_list = list(data_cursor)

    response_data = {
        "contents": data_list
    }

    return JsonResponse(response_data, safe=False)
