from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from pymongo import MongoClient
import fitz  # PyMuPDF
import logging
import io
from typing import List, Optional, Tuple, Dict, Union
from flask import Flask, jsonify, request
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pymongo import MongoClient
import django
import os
import openpyxl
import certifi
import pandas as pd
from django.http import JsonResponse

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  
import nltk
from nltk.corpus import stopwords  # stopwords import
# Download stopwords punkt(if not already done)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

import openai

# 발급받은 API 키 설정
client = openai.Client(api_key = "OPENAI_API_KEY")


# Set the settings module for the Django project
os.environ.setdefault("DJANGO_SETTINGS_MODULE","TalkStockReport.settings")

# Initialize Django
django.setup()

#check initialization Django
#print("Django initialized successfully.")

logging.getLogger("urllib3").setLevel(logging.WARNING)


# PDF 텍스트 추출 클래스
class PDFTextExtractor:
    def __init__(self, pdf_source: Union[str, io.BytesIO]):
        self.pdf_source = pdf_source
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def extract_text(
            self,
            page_number: int,
            rect_coordinates: Tuple[float, float, float, float],
            stop_keywords: Optional[List[Dict[str, Union[str, bool]]]] = None
    ) -> str:
        try:
            # PDF 열기 (URL 또는 로컬 파일 경로)
            if isinstance(self.pdf_source, str):
                doc = fitz.open(self.pdf_source)
            else:
                doc = fitz.open("pdf", self.pdf_source)

            # 페이지 가져오기
            page = doc[page_number]

            # 영역 지정
            rect = fitz.Rect(*rect_coordinates)

            # 해당 영역에서 텍스트 추출
            text = page.get_text("text", clip=rect)

            # 문서 닫기
            doc.close()

            # 줄 단위로 분리
            lines = text.split('\n')

            # 중단 키워드 처리
            if stop_keywords:
                for keyword_config in stop_keywords:
                    keyword = keyword_config.get('keyword', '')
                    exclude_two_before = keyword_config.get('exclude_keyword_two_before_line', False)

                    for i, line in enumerate(lines):
                        if keyword in line.strip():
                            if exclude_two_before:
                                # 키워드 2줄 전까지 텍스트 추출
                                cut_index = max(0, i - 2)
                                text = '\n'.join(lines[:cut_index])
                            else:
                                # 키워드 줄 전까지 텍스트 추출
                                text = '\n'.join(lines[:i])

                            break

            return text.strip()

        except Exception as e:
            self.logger.error(f"PDF 텍스트 추출 중 오류 발생: {e}")
            return ""


# 증권사별 기본 설정 딕셔너리 (PDF 추출 설정)
SECURITIES_CONFIGS = {
    "SK증권": {
        "page_num": 0,
        "coordinates": (180, 150, 700, 690),
        "stop_keywords": []
    },
    "교보증권": {
        "page_num": 0,
        "coordinates": (190, 240, 700, 655),
        "stop_keywords": []
    },
    "나이스디앤비": {
        "page_num": 1,
        "coordinates": (180, 200, 700, 682),
        "stop_keywords": []
    },
    "메리츠증권": {
        "page_num": 0,
        "coordinates": (210, 200, 700, 700),
        "stop_keywords": [{"keyword": "EPS (원)", "exclude_keyword_two_before_line": False}]
    },
    "미래에셋증권": {
        "page_num": 0,
        "coordinates": (200, 170, 700, 650),
        "stop_keywords": []
    },
    "삼성증권": {
        "page_num": 0,
        "coordinates": (200, 170, 700, 700),
        "stop_keywords": [{"keyword": "분기 실적", "exclude_keyword_two_before_line": False}]
    },
    "신한투자증권": {
        "page_num": 0,
        "coordinates": (20, 190, 350, 560),
        "stop_keywords": []
    },
    "유안타증권": {
        "page_num": 0,
        "coordinates": (30, 180, 400, 650),
        "stop_keywords": [{"keyword": "Forecasts and valuations (K-IFRS 연결)", "exclude_keyword_two_before_line": False}]
    },
    "유진투자증권": {
        "page_num": 0,
        "coordinates": (30, 275, 680, 700),
        "stop_keywords": [{"keyword": "시가총액(십억원)", "exclude_keyword_two_before_line": True}]
    },
    "키움증권": {
        "page_num": 0,
        "coordinates": (220, 190, 700, 850),
        "stop_keywords": []
    },
    "하나증권": {
        "page_num": 0,
        "coordinates": (179, 140, 700, 850),
        "stop_keywords": []
    },
    "한국IR협의회": {
        "page_num": 1,
        "coordinates": (40, 200, 370, 700),
        "stop_keywords": [{"keyword": "Forecast earnings & Valuation", "exclude_keyword_two_before_line": False}]
    },
    "한국기술신용평가(주)": {
        "page_num": 1,
        "coordinates": (180, 200, 700, 682),
        "stop_keywords": []
    },
    "한화투자증권": {
        "page_num": 0,
        "coordinates": (240, 220, 680, 800),
        "stop_keywords": []
    },
}


def download_and_process_pdf(pdf_url: str, company: str) -> str:
    # 증권사 설정 가져오기
    config = SECURITIES_CONFIGS.get(company)

    # 설정이 없는 경우 처리
    if not config:
        print(f"{company}에 대한 설정이 없습니다.")
        return ""

    try:
        # 1. PDF 다운로드
        response = requests.get(pdf_url)
        response.raise_for_status()

        # 2. 메모리 버퍼에 PDF 로드
        pdf_buffer = io.BytesIO(response.content)

        try:
            # 3. PDF 처리
            extractor = PDFTextExtractor(pdf_buffer)
            text = extractor.extract_text(
                page_number=config.get('page_num', 1),
                rect_coordinates=config.get('coordinates', (0, 0, 0, 0)),
                stop_keywords=config.get('stop_keywords', [])
            )

            return text

        finally:
            # 4. 메모리 버퍼 닫기
            pdf_buffer.close()

    except requests.exceptions.RequestException as e:
        print(f"PDF 다운로드 중 오류 발생: {e}")
        return ""
    except Exception as e:
        print(f"PDF 처리 중 오류 발생: {e}")
        return ""


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


# Function to fetch detailed content for stock and industry reports
def fetch_report_details(detail_url):
    response = requests.get(detail_url)
    response.encoding = 'euc-kr'
    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        content_div = soup.find("div",
                                style="width:555px;height:100% clear:both; text-align: justify; overflow-x: auto;padding: 20px 0pt 30px;font-size:9pt;line-height:160%; color:#000000;")
        paragraphs = content_div.find_all("p")
        if len(paragraphs) > 2:
            content = paragraphs[2].get_text(" ", strip=True)
            if len(paragraphs) == 4:
                content += " " + paragraphs[3].get_text(" ", strip=True)
        elif len(paragraphs) > 1:
            content = paragraphs[1].get_text(" ", strip=True)
        elif len(paragraphs) == 1:
            content = paragraphs[0].get_text(" ", strip=True)
        else:
            content = "내용이 없습니다."
    except AttributeError:
        content = "내용을 가져올 수 없습니다."
    return content

# 모든 업종의 개별 페이지 링크 
def fetch_industry_data():
    base_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    response = requests.get(base_url)
    response.encoding = "euc-kr"
    soup = BeautifulSoup(response.text, "html.parser")

    industry_data = {}  # 업종명 -> 개별 페이지 URL

    table = soup.find("table", {"class": "type_1"})
    if table:
        for row in table.find_all("tr")[2:]:  # 첫 두 줄(헤더) 제외
            cols = row.find_all("td")
            if len(cols) < 3:
                continue

            industry_name = cols[0].text.strip()
            industry_link = cols[0].find("a")["href"]  # 업종 상세 페이지 링크
            full_url = f"https://finance.naver.com{industry_link}"  # 절대 URL로 변환
            
            # 각 업종의 종목 리스트 가져오기
            industry_data[industry_name] = fetch_stocks_by_industry(full_url)

    return industry_data  # {업종명: URL}


# 특정 업종의 종목 리스트를 가져옴
def fetch_stocks_by_industry(industry_url):
   
    # 네이버 요청 차단 가능성 대비
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(industry_url, headers=headers)
    response.encoding = "euc-kr"
    soup = BeautifulSoup(response.text, "html.parser")
    
    stock_list = []
    
    table = soup.find("table", {"class": "type_5"})
    if not table:
        return []  # 테이블이 없으면 빈 리스트 반환
    
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) == 0:  # 데이터가 부족한 행 제외
            continue

        # 종목명 추출
        #name_area = cols[0].find("div", class_="name_area")
        company_name_tag = cols[0].find("a")
        if company_name_tag is None:
            continue

        company_name = company_name_tag.text.strip() 
        stock_list.append(company_name)

    return stock_list  # 종목 리스트 반환


##### 종목분석 리포트에 대한 크롤링 #####
def fetch_stock_reports(category_name, category_url, pages, industry_data):
    reports = []
    
    for page in range(1, pages + 1):
        url = f"{category_url}?&page={page}"
        response = requests.get(url)
        response.encoding = 'euc-kr'
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table", {"class": "type_1"})

        if not table:
            print(f"Table not found for URL: {url}")
            continue

        for row in table.find_all("tr")[2:]:
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            item_name = cols[0].text.strip() 
            title = cols[1].text.strip()
            detail_link = cols[1].find("a")["href"]
            detail_url = f"https://finance.naver.com/research/{detail_link}" if not detail_link.startswith(
                "http") else detail_link
            company = cols[2].text.strip() # 증권사

            # SECURITIES_CONFIGS에서 증권사 필터링
            if company not in SECURITIES_CONFIGS:
                continue

            pdf_link_tag = cols[3].find("a")
            pdf_url = pdf_link_tag["href"] if pdf_link_tag and "href" in pdf_link_tag.attrs else "PDF 없음"
            date = cols[4].text.strip()
            views = cols[5].text.strip()

            # PDF 콘텐츠 추출
            pdf_content = "" if pdf_url == "PDF 없음" else download_and_process_pdf(pdf_url,company)
            report_content = fetch_report_details(detail_url)

            # 종목명을 기반으로 업종 찾기 + 리스트를 문자열로 변환
            industry_list = [
                industry for industry, stock_list in industry_data.items() if item_name in stock_list
            ]
            industry_value = industry_list[0] if industry_list else "Unknown"

            reports.append({
                'Category': category_name,
                '종목명': item_name,
                '업종': industry_value,
                'Title': title,
                '증권사': company,
                'PDF URL': pdf_url,
                '작성일': date,
                'Views': views,
                'Content': report_content,
                'PDF Content': pdf_content,
            })
    return reports

##### 산업분석 리포트에 대한 크롤링 ##### 
def fetch_industry_reports(category_name, category_url, pages):
    reports = []
    for page in range(1, pages + 1):
        url = f"{category_url}?&page={page}"
        response = requests.get(url)
        response.encoding = 'euc-kr'
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find("table", {"class": "type_1"})

        if not table:
            print(f"Table not found for URL: {url}")
            continue

        for row in table.find_all("tr")[2:]:
            cols = row.find_all("td")
            if len(cols) < 5:
                continue
            
            item_name = cols[0].text.strip() # 여기서 item_name은 업종명이므로 '업종'에 저장
            title = cols[1].text.strip()
            detail_link = cols[1].find("a")["href"]
            detail_url = f"https://finance.naver.com/research/{detail_link}" if not detail_link.startswith(
                "http") else detail_link
            company = cols[2].text.strip() # 증권사

            # SECURITIES_CONFIGS에서 증권사 필터링
            if company not in SECURITIES_CONFIGS:
                continue

            pdf_link_tag = cols[3].find("a")
            pdf_url = pdf_link_tag["href"] if pdf_link_tag and "href" in pdf_link_tag.attrs else "PDF 없음"
            date = cols[4].text.strip()
            views = cols[5].text.strip()

            # PDF 콘텐츠 추출
            pdf_content = "" if pdf_url == "PDF 없음" else download_and_process_pdf(pdf_url,company)
            report_content = fetch_report_details(detail_url)

            reports.append({
                'Category': category_name,
                '종목명': "",  # 산업 분석 리포트는 종목명을 비워둠
                '업종': item_name,  # item_name을 업종으로 저장
                'Title': title,
                '증권사': company,
                'PDF URL': pdf_url,
                '작성일': date,
                'Views': views,
                'Content': report_content,
                'PDF Content': pdf_content,
            })

    return reports


##### 시황 정보, 투자 정보, 경제 분석, 채권 분석 리포트에 대한 크롤링 #####
def fetch_other_reports(category_name, category_url, pages):
    reports = []
    for page in range(1, pages + 1):
        url = f"{category_url}?&page={page}"
        response = requests.get(url)
        response.encoding = 'euc-kr'
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find("table", {"class": "type_1"})
        if not table:
            print(f"Table not found for URL: {url}")
            continue

        for row in table.find_all("tr")[2:]:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            title_tag = cols[0].find("a")
            title = title_tag.text.strip()
            detail_link = title_tag["href"]
            detail_url = f"https://finance.naver.com/research/{detail_link}"
            company = cols[1].text.strip()

            pdf_link_tag = cols[2].find("a")
            pdf_url = pdf_link_tag["href"] if pdf_link_tag and "href" in pdf_link_tag.attrs else "PDF 없음"
            date = cols[3].text.strip()
            views = cols[4].text.strip()

            # 상세 페이지 데이터 가져오기
            report_content = fetch_report_details(detail_url)

            reports.append({
                'Category': category_name,
                'Title': title,
                '증권사': company,
                'PDF URL': pdf_url,
                '작성일': date,
                'Views': views,
                'Content': report_content,
            })
    return reports


##### 전체 보고서 크롤링 및 저장 함수 #####
def fetch_all_reports(pages=1):
    global df
    base_url = "https://finance.naver.com/research/"
    categories = {
        '종목분석 리포트': f"{base_url}company_list.naver",
        '산업분석 리포트': f"{base_url}industry_list.naver",
        '시황정보 리포트': f"{base_url}market_info_list.naver",
        '투자정보 리포트': f"{base_url}invest_list.naver",
        '경제분석 리포트': f"{base_url}economy_list.naver",
        '채권분석 리포트': f"{base_url}debenture_list.naver",
    }
    # 업종별 종목 매핑 가져오기
    industry_mapping = fetch_industry_data()

    all_reports = []

    for category_name, category_url in categories.items():
        if category_name == '종목분석 리포트':
            reports = fetch_stock_reports(category_name, category_url, pages, industry_mapping)
        elif category_name == '산업분석 리포트':
            reports = fetch_industry_reports(category_name, category_url, pages)
        else:
            reports = fetch_other_reports(category_name, category_url, pages)

        all_reports.extend(reports)

    # Save all reports to DataFrame
    df = pd.DataFrame(all_reports)
    df['Content'] = df['Content'].apply(lambda x: f"'{x}'" if pd.notnull(x) else x)
    df.to_excel('all_reports.xlsx', index=False)
    print("All reports saved to all_reports.xlsx")

# OpenAI API를 사용한 요약 함수
def summarize_text_with_gpt(text):
    """
    Uses OpenAI's GPT API to summarize a given text.

    Args:
        text (str): The input text to summarize.

    Returns:
        str: The summarized text.
    """
    if not text or text.strip() == "":
        return "No content available for summarization."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 한국어 금융 보고서를 요약하는 AI입니다. 절대 영어를 사용하지 마세요. 모든 요약을 반드시 한국어로 작성해야 합니다"},
                {"role": "user", "content": f"다음 금융 보고서를 한국어로 3문장으로 요약하세요. 절대 영어를 사용하지 마세요:\n{text}"}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"    

# all report를 가져오기
fetch_all_reports(pages=2)
print(df)

df['Content'] = df['Content'].apply(lambda x: x.replace("\n", "") if pd.notnull(x) else x)
df['PDF Content'] = df['PDF Content'].apply(lambda x: x.replace("\n", "") if pd.notnull(x) else x)

df_cleaned = df.dropna(subset=['PDF Content'])

# Apply the summarization function to the "PDF Content" column
df['Summary'] = df['PDF Content'].astype(str).apply(lambda x: summarize_text_with_gpt(x) if pd.notnull(x) else "")

# Save the updated DataFrame to a new Excel fileç
output_file_path = "all_reports_with_summary.xlsx"
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
    client = MongoClient("mongodb://mongodb:27017/")
    db = client['report_database']
    collection = db['reports']

    # 작성일 기준으로 정렬하여 최신 데이터 반환
    data_cursor = collection.find({}, {"_id": 0}).sort("작성일", -1)  # 작성일 기준 내림차순 정렬
    data_list = list(data_cursor)

    response_data = {
        "contents": data_list
    }

    return JsonResponse(response_data, safe=False)
