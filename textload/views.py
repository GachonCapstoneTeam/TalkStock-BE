import os
import django
import logging
import pandas as pd
from collections import defaultdict
from rest_framework.decorators import api_view
from rest_framework.response import Response
from pymongo import MongoClient
import openai
from django.http import JsonResponse
from . import pdfToText as pdf #pdf파일을 크롤링
from . import crawling as crawling #네이버리포트를 크롤링
from . import keywordExtraction as kw #키워드 추출
from . import keywordEvaluation as eval #키워드 평가
import ast
from  .evaluation import evaluate_all_keywords
from statistics import count_reports_by_industry
# Django 환경설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE","TalkStockReport.settings")
django.setup()

# 로깅 설정
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 발급받은 API 키 설정
client = openai.Client(api_key = "OPENAI_API_KEY")

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


#유효한 문자만 남기기 위한 필터링
def remove_illegal_chars(text):
    if  isinstance(text, str):
                return ''.join(c for c in text if c >= " " and c not in ['\x7f', '\x9f'])  #허용되지 않은 unicode 문자 제거
    return text

##### 전체 보고서 크롤링 및 저장 함수 #####
def fetch_all_reports(pages=1):
    global df
    base_url = "https://finance.naver.com/research/"
    categories = {
        '종목분석 리포트': f"{base_url}company_list.naver",
        '산업분석 리포트': f"{base_url}industry_list.naver",
    }
    # 업종별 종목 매핑 가져오기
    industry_mapping = crawling.fetch_industry_data()

    all_reports = []

    for category_name, category_url in categories.items():
        if category_name == '종목분석 리포트':
            reports = crawling.fetch_stock_reports(category_name, category_url, pages, industry_mapping)
        elif category_name == '산업분석 리포트':
            reports = crawling.fetch_industry_reports(category_name, category_url, pages)
        else:
            reports = crawling.fetch_other_reports(category_name, category_url, pages)

        all_reports.extend(reports)

    # 리포트 데이터프레임 생성
    df = pd.DataFrame(all_reports)
    df = df.applymap(remove_illegal_chars)
    df['Content'] = df['Content'].apply(lambda x: f"'{x}'" if pd.notnull(x) else x)
    df.to_excel('all_reports.xlsx', index=False)
    print("All reports saved : all_reports.xlsx")

# # OpenAI API를 사용한 요약 함수
# def summarize_text_with_gpt(text):

#     if not text or text.strip() == "":
#         return "No content available for summarization."

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o mini",
#             messages=[
#                 {"role": "system", "content": "당신은 한국어 금융 보고서를 1문장으로 요약하는 AI야. 절대 영어를 사용하지 마. 모든 요약을 반드시 한국어로 작성해"},
#                 {"role": "user", "content": f"다음 금융 보고서를 한국어로 1문장으로 요약해줘. 절대 영어를 사용하지 마:\n{text}"}
#             ],
#             max_tokens=100
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error: {e}"    

# 네이버 리포트 크롤링할 페이지 수 지정
all_reports = fetch_all_reports(pages=5)

# 업종별 키워드 모음 생성
industry_kewords_map = defaultdict(list)

df['Content'] = df['Content'].apply(lambda x: x.replace("\n", "") if pd.notnull(x) else x)
df['PDF Content'] = df['PDF Content'].apply(lambda x: x.replace("\n", "") if pd.notnull(x) else x)
df['TFIDF'] = df['Content'].astype(str).apply(lambda x: kw.extract_keywords_tfidf(x, top_n = 5))
df['KRWordRank'] = df['PDF Content'].astype(str).apply(lambda x: kw.extract_keywords_krwordrank(x, top_n = 5))
df['keybert'] = df['Content'].astype(str).apply(lambda x: kw.extract_keywords_keybert(x, top_n = 5))
df['Summary'] = df['Content'].astype(str).apply(lambda x: kw.summarize_text_kobart(x))

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

### 업종별 키워드 저장
output_file_path = "all_reports_keywords.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Summarization complete! The updated file is saved as: {output_file_path}")

# 딕셔너리 형태로 변환
data_to_save = df.to_dict('records')

# MongoDB에 데이터 저장
insert_data_into_mongo(data_to_save)
print("데이터가 MongoDB에 저장되었습니다!!")


# 문자열 형태의 리스트를 진짜 리스트로 변환
def safe_eval_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except:
        return []
# === 키워드 평가 호출 ===
documents = df['PDF Content'].astype(str).tolist()
keywords_tfidf = df['TFIDF'].apply(safe_eval_list).tolist()
keywords_keybert = df['keybert'].apply(safe_eval_list).tolist()
keywords_krwordrank = df['KRWordRank'].apply(safe_eval_list).tolist()

all_keywords = {
    "TF-IDF": keywords_tfidf,
    "KeyBERT": keywords_keybert,
    "KRWordRank": keywords_krwordrank
}
results_df = evaluate_all_keywords(documents, all_keywords)

# 점수 확인
print(results_df[['TF-IDF_Score', 'KeyBERT_Score', 'KRWordRank_Score']].head())

### 업종별 리포트 수 세기 ###
# 엑셀 파일 불러오기 또는 df 생성 이후
industry_df = count_reports_by_industry(df)
print(industry_df.head())
industry_df.to_excel("industry_report_count.xlsx", index=False)

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

