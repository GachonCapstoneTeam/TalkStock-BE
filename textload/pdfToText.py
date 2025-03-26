import fitz  # PyMuPDF
import logging
import requests
import io
from typing import List, Optional, Tuple, Dict, Union

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

# 증권사별 기본 설정 딕셔너리 (대충 채움)
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

#1페이지만 추출
def download_and_process_pdf(pdf_url: str, securities_firm: str) -> str:

    try:
        # 1. PDF 다운로드
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # 2. 메모리 버퍼에 PDF 로드
        pdf_buffer = io.BytesIO(response.content)
        
        try:
            # 3. PDF 처리
            config = SECURITIES_CONFIGS.get(securities_firm)
            if not config:
                print(f"{securities_firm}에 대한 설정이 없습니다.")
                return ""
            
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

#1페이지 추출후 나머지 페이지도 추출
def download_and_process_pdf2(pdf_url: str, securities_firm: str) -> str:
    try:
        # 1. PDF 다운로드
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # 2. 메모리 버퍼에 PDF 로드
        pdf_buffer = io.BytesIO(response.content)
        
        try:
            # PDF 문서 열기
            doc = fitz.open(stream=pdf_buffer, filetype="pdf")
            total_pages = len(doc)  # 전체 페이지 수 가져오기

            # 설정값 가져오기
            config = SECURITIES_CONFIGS.get(securities_firm)
            if not config:
                print(f"{securities_firm}에 대한 설정이 없습니다.")
                return ""

            extractor = PDFTextExtractor(pdf_buffer)

            # 첫 페이지 (특정 영역만 추출)
            first_page_text = extractor.extract_text(
                page_number=config.get('page_num', 1),
                rect_coordinates=config.get('coordinates', (0, 0, 0, 0)),
                stop_keywords=config.get('stop_keywords', [])
            )

            # 2페이지 이후 (전체 페이지 텍스트 추출)
            additional_text = ""
            for page_number in range(2, total_pages + 1):
                # 해당 페이지의 전체 크기 가져오기
                page_rect = doc[page_number - 1].rect  # PyMuPDF는 0-based index 사용
                full_page_text = extractor.extract_text(
                    page_number=page_number,
                    rect_coordinates=(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)  # 전체 영역 사용
                )
                additional_text += full_page_text + "\n"

            return first_page_text + "\n" + additional_text

        finally:
            # 4. 메모리 버퍼 닫기
            pdf_buffer.close()
    
    except requests.exceptions.RequestException as e:
        print(f"PDF 다운로드 중 오류 발생: {e}")
        return ""
    except Exception as e:
        print(f"PDF 처리 중 오류 발생: {e}")
        return ""
