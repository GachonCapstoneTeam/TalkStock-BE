import requests
from bs4 import BeautifulSoup
from . import pdfToText as pdf        
# from pdfToText import download_and_process_pdf2 # PDF 함수 사용을 위해
# from pdfToText import SECURITIES_CONFIGS # 증권사 설정도 pdfToText에서 불러오기

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
            
            # 종목명 및 종목 코드 추출
            stock_link = cols[0].find("a", class_="stock_item")
            if stock_link:
                item_name = stock_link.text.strip()  # 종목명
                code = stock_link["href"].split("=")[-1]  # 종목 코드
            else:
                item_name = cols[0].text.strip()
                code = None  # 코드 정보 없음
            # item_name = cols[0].text.strip() 
            title = cols[1].text.strip()
            detail_link = cols[1].find("a")["href"]
            detail_url = f"https://finance.naver.com/research/{detail_link}" if not detail_link.startswith(
                "http") else detail_link
            company = cols[2].text.strip() # 증권사

            # SECURITIES_CONFIGS에서 증권사 필터링
            if company not in pdf.SECURITIES_CONFIGS:
                continue

            pdf_link_tag = cols[3].find("a")
            pdf_url = pdf_link_tag["href"] if pdf_link_tag and "href" in pdf_link_tag.attrs else "PDF 없음"
            date = cols[4].text.strip()
            views = cols[5].text.strip()

            # PDF 콘텐츠 추출
            pdf_content = "" if pdf_url == "PDF 없음" else pdf.download_and_process_pdf2(pdf_url,company)
            report_content = fetch_report_details(detail_url)

            # 종목명을 기반으로 업종 찾기 + 리스트를 문자열로 변환
            industry_list = [
                industry for industry, stock_list in industry_data.items() if item_name in stock_list
            ]
            industry_value = industry_list[0] if industry_list else "Unknown"

            reports.append({
                'Category': category_name,
                '종목명': item_name,
                '코드' : code ,
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
            if company not in pdf.SECURITIES_CONFIGS:
                continue

            pdf_link_tag = cols[3].find("a")
            pdf_url = pdf_link_tag["href"] if pdf_link_tag and "href" in pdf_link_tag.attrs else "PDF 없음"
            date = cols[4].text.strip()
            views = cols[5].text.strip()

            # PDF 콘텐츠 추출
            pdf_content = "" if pdf_url == "PDF 없음" else pdf.download_and_process_pdf2(pdf_url,company)
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
