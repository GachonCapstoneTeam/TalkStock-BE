from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from pymongo import MongoClient


# Create your views here.

from flask import Flask, jsonify, request
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pymongo import MongoClient


def connect_to_mongo():
    client = MongoClient("mongodb://mongodb:27017/")  # 연결 문자열을 필요에 따라 변경
    db = client["report_database"]  # 데이터베이스 이름
    return db


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


# Function to crawl stock analysis and industry reports
def fetch_stock_and_industry_reports(category_name, category_url, pages):
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

            itemName = cols[0].text.strip()
            title = cols[1].text.strip()
            detail_link = cols[1].find("a")["href"]
            detail_url = f"https://finance.naver.com/research/{detail_link}" if not detail_link.startswith(
                "http") else detail_link
            company = cols[2].text.strip()
            pdf_link_tag = cols[3].find("a")
            pdf_url = pdf_link_tag["href"] if pdf_link_tag and "href" in pdf_link_tag.attrs else "PDF 없음"
            date = cols[4].text.strip()
            views = cols[5].text.strip()

            report_content = fetch_report_details(detail_url)
            reports.append({
                'Category': category_name,
                # '종목명': itemName,
                'Title': title,
                '증권사': company,
                'PDF URL': pdf_url,
                '작성일': date,
                'Views': views,
                'Content': report_content,
            })
    return reports


# Function to fetch market, investment, economy, debenture reports
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


# Main function to crawl all reports
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
    all_reports = []

    for category_name, category_url in categories.items():
        if category_name in ['종목분석 리포트', '산업분석 리포트']:
            reports = fetch_stock_and_industry_reports(category_name, category_url, pages)
        else:
            reports = fetch_other_reports(category_name, category_url, pages)
        all_reports.extend(reports)

    # Save all reports to DataFrame
    df = pd.DataFrame(all_reports)
    df['Content'] = df['Content'].apply(lambda x: f"'{x}'" if pd.notnull(x) else x)
    df.to_excel('all_reports.xlsx', index=False)
    print("All reports saved to all_reports.xlsx")


# Run the function to fetch all reports
fetch_all_reports(pages=2)
print(df)

# Convert dataFrame to list of dictionaries
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