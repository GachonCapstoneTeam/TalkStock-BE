import fitz  # PyMuPDF
import logging
import requests
import io
from typing import List, Optional, Tuple, Dict, Union

# 로거 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # INFO 레벨로 변경 (DEBUG는 너무 상세할 수 있음)
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self):
        pass

    def extract_text_from_page(
        self,
        page: 'fitz.Page',
        rect_coordinates: Tuple[float, float, float, float],
        stop_keywords: Optional[List[Dict[str, Union[str, bool]]]] = None
    ) -> str:
        try:
            rect = fitz.Rect(*rect_coordinates)
            text = page.get_text("text", clip=rect)
            lines = text.split('\n')

            #logger.debug(f"--- 페이지 {page.number} extract_text_from_page 시작 ---")
            #logger.debug(f"  전달된 stop_keywords: {stop_keywords}")

            if not stop_keywords:
                #logger.debug(f"페이지 {page.number}: Stop keyword 설정 없음. 전체 반환.")
                return text.strip()

            earliest_stop_line_index = -1
            final_exclude_two_before = False
            found_keyword_text = "" # 로깅용

            #logger.debug(f"페이지 {page.number}: 라인 순회 시작 (총 {len(lines)} 라인)")
            for i, line in enumerate(lines):
                line_stripped = line.strip()

                if not line_stripped:
                     continue

                for keyword_config in stop_keywords:
                    keyword = keyword_config.get('keyword', '')
                    exclude_two_before = keyword_config.get('exclude_keyword_two_before_line', False)

                    if keyword:
                        # --- *** 수정된 비교 로직 시작 *** ---
                        # 1. 키워드와 라인 정리 (소문자 변환, 앞뒤 공백 제거)
                        cleaned_keyword = keyword.strip().lower()
                        cleaned_line = line_stripped.lower()

                        # 2. 양방향 포함 관계 확인
                        #    - 설정 키워드가 라인에 포함되는가? (기존 방식)
                        #    - 또는, 라인이 설정 키워드에 포함되는가? (라인이 키워드의 일부인 경우)
                        match_found = (cleaned_keyword in cleaned_line) or \
                                      (cleaned_line in cleaned_keyword)

                        # 디버깅 로그 추가: 어떤 조건으로 매칭되었는지 확인
                        match_reason = ""
                        if cleaned_keyword in cleaned_line:
                            match_reason = "(키워드가 라인에 포함)"
                        elif cleaned_line in cleaned_keyword:
                            match_reason = "(라인이 키워드에 포함)"

                        #logger.debug(f"    Line {i}: 비교: Ln='{cleaned_line}' | Kw='{cleaned_keyword}' | 포함됨?: {match_found} {match_reason}")
                        # --- *** 수정된 비교 로직 끝 *** ---

                        if match_found:
                            # 길이를 이용한 추가 필터링 (선택 사항): 너무 짧은 라인이 긴 키워드에 포함되는 것을 방지
                            if cleaned_line in cleaned_keyword and len(cleaned_line) < 5: # 예: 5글자 미만 라인은 무시
                                #logger.debug(f"      라인이 키워드에 포함되지만 너무 짧음({len(cleaned_line)} < 5). 무시.")
                                continue # 다음 키워드 또는 다음 라인 검사로 넘어감

                            #logger.info(f"  >>> 페이지 {page.number}, 라인 {i}: 키워드 '{keyword}' 일부 또는 전체 포함 발견 {match_reason}! in '{line_stripped}'")
                            if earliest_stop_line_index == -1 or i < earliest_stop_line_index:
                                #logger.info(f"      (갱신) earliest_stop_line_index: {i}, exclude_two_before: {exclude_two_before}")
                                earliest_stop_line_index = i
                                final_exclude_two_before = exclude_two_before
                                found_keyword_text = keyword
                            break # 현재 라인에서 키워드 찾았으면 다음 라인으로

            #logger.debug(f"--- 페이지 {page.number} 텍스트 검사 종료 (earliest_stop_line_index: {earliest_stop_line_index}) ---")

            if earliest_stop_line_index != -1:
                cut_index = earliest_stop_line_index
                if final_exclude_two_before:
                    cut_index = max(0, earliest_stop_line_index - 2)

                result_text = '\n'.join(lines[:cut_index])
                #logger.info(f"페이지 {page.number}: Stop keyword '{found_keyword_text}' 일부 또는 전체 포함 기준으로 텍스트 잘림 (cut_index: {cut_index}).")
                return result_text.strip()
            else:
                #logger.info(f"페이지 {page.number}: Stop keyword 일부 또는 전체 포함 라인 찾지 못함 (검색 키워드: {stop_keywords}). 전체 텍스트 반환.")
                return text.strip()

        except Exception as e:
            #logger.error(f"PDF 페이지({page.number}) 텍스트 추출 중 오류 발생: {e}")
            return ""

# 증권사별 기본 설정 딕셔너리 (이전과 동일)
SECURITIES_CONFIGS = {
    "SK증권": { "page_num": 0, "coordinates": (180, 150, 700, 690), "stop_keywords": [] },
    "교보증권": { "page_num": 0, "coordinates": (190, 240, 700, 655), "stop_keywords": [] },
    "나이스디앤비": { "page_num": 1, "coordinates": (180, 200, 700, 682), "stop_keywords": [] },
    "메리츠증권": { "page_num": 0, "coordinates": (210, 200, 700, 700), "stop_keywords": [{"keyword": "EPS (원)", "exclude_keyword_two_before_line": False}] },
    "미래에셋증권": { "page_num": 0, "coordinates": (200, 170, 700, 650), "stop_keywords": [] },
    "삼성증권": { "page_num": 0, "coordinates": (200, 170, 700, 700), "stop_keywords": [{"keyword": "분기 실적", "exclude_keyword_two_before_line": False}] },
    "신한투자증권": { "page_num": 0, "coordinates": (20, 190, 350, 560), "stop_keywords": [] },
    "유안타증권": { "page_num": 0, "coordinates": (30, 180, 400, 650), "stop_keywords": [{"keyword": "Forecasts and valuations (K-IFRS 연결)", "exclude_keyword_two_before_line": False}] },
    "유진투자증권": { "page_num": 0, "coordinates": (30, 275, 680, 700), "stop_keywords": [{"keyword": "시가총액(십억원)", "exclude_keyword_two_before_line": True}] },
    "키움증권": { "page_num": 0, "coordinates": (220, 190, 700, 850), "stop_keywords": [] },
    "하나증권": { "page_num": 0, "coordinates": (179, 140, 700, 850), "stop_keywords": [] },
    "한국IR협의회": { "page_num": 1, "coordinates": (40, 200, 370, 700), "stop_keywords": [{"keyword": "Forecast earnings & Valuation", "exclude_keyword_two_before_line": False}] },
    "한국기술신용평가(주)": { "page_num": 1, "coordinates": (180, 200, 700, 682), "stop_keywords": [] },
    "한화투자증권": { "page_num": 0, "coordinates": (240, 220, 680, 800), "stop_keywords": [] },
    # 필요시 다른 증권사 추가
}

# 1페이지만 추출
def download_and_process_pdf(pdf_url: str, securities_firm: str) -> str:
    doc = None
    pdf_buffer = None
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        pdf_buffer = io.BytesIO(response.content)

        try:
            doc = fitz.open(stream=pdf_buffer, filetype="pdf")
        except RuntimeError as e:
            logger.error(f"fitz.open 실패 ({pdf_url}): {e}.")
            return ""
        except Exception as e:
             logger.error(f"fitz.open 중 예기치 못한 오류 ({pdf_url}): {e}")
             return ""

        total_pages = len(doc)
        if total_pages == 0:
            logger.warning(f"PDF에 페이지가 없습니다 ({pdf_url}).")
            return ""

        config = SECURITIES_CONFIGS.get(securities_firm)
        if not config:
            logger.warning(f"{securities_firm} 설정 없어 첫 페이지 전체 추출 ({pdf_url}).")
            page_index = 0
            coordinates = doc[0].rect # 전체 영역
            stop_keywords = []
        else:
            page_index = config.get('page_num', 0)
            coordinates = config.get('coordinates', (0,0,0,0))
            stop_keywords = config.get('stop_keywords', [])

            if not (0 <= page_index < total_pages):
                logger.warning(f"설정된 페이지 번호 {page_index}(0-based) 유효하지 않음(총 {total_pages}p). 첫 페이지 전체 추출 ({pdf_url}).")
                page_index = 0
                coordinates = doc[0].rect

            if coordinates == (0,0,0,0): # 좌표 설정 없으면 전체 영역
                 logger.warning(f"{securities_firm} 좌표 설정 없어 {page_index} 페이지 전체 추출 ({pdf_url}).")
                 coordinates = doc[page_index].rect

        extractor = PDFTextExtractor()
        page = doc.load_page(page_index)

        text = extractor.extract_text_from_page(
            page=page,
            rect_coordinates=coordinates,
            stop_keywords=stop_keywords
        )
        return text

    except requests.exceptions.RequestException as e:
        logger.error(f"PDF 다운로드 오류 ({pdf_url}): {e}")
        return ""
    except Exception as e:
        logger.error(f"PDF 처리 중 예기치 못한 오류 ({pdf_url}): {e}")
        return ""
    finally:
        if doc:
            doc.close()
        if pdf_buffer:
            pdf_buffer.close()

# --- download_and_process_pdf2 함수 수정 ---
def download_and_process_pdf2(pdf_url: str, securities_firm: str) -> str:
    doc = None
    pdf_buffer = None
    try:
        # 1. PDF 다운로드
        response = requests.get(pdf_url, timeout=30) # 타임아웃 추가
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        pdf_buffer = io.BytesIO(response.content)

        # 2. PDF 문서 열기 (한 번만)
        try:
            doc = fitz.open(stream=pdf_buffer, filetype="pdf")
        except RuntimeError as e: # fitz.open 실패 시 RuntimeError 발생 가능
            #logger.error(f"fitz.open 실패 ({pdf_url}): {e}. 손상된 파일이거나 지원하지 않는 형식일 수 있습니다.")
            return ""
        except Exception as e: # 다른 예기치 못한 오류
             #logger.error(f"fitz.open 중 예기치 못한 오류 ({pdf_url}): {e}")
             return ""

        total_pages = len(doc)
        if total_pages == 0:
            #logger.warning(f"PDF에 페이지가 없습니다 ({pdf_url}).")
            return ""

        # 설정값 가져오기
        config = SECURITIES_CONFIGS.get(securities_firm)
        if not config:
            #logger.warning(f"{securities_firm}에 대한 설정이 없어 전체 페이지 텍스트를 추출합니다 ({pdf_url}).")
            # 설정 없을 시 전체 페이지 추출
            first_page_index = -1 # 특정 페이지 설정 없음을 의미
            coordinates = (0, 0, 0, 0)
            stop_keywords = []
        else:
            # config의 page_num은 0-based index 라고 가정
            first_page_index = config.get('page_num', 0)
            coordinates = config.get('coordinates', (0, 0, 0, 0))
            stop_keywords = config.get('stop_keywords', [])

            # 페이지 인덱스 유효성 검사
            if not (0 <= first_page_index < total_pages):
                 #logger.warning(f"설정된 페이지 번호 {first_page_index}(0-based)가 유효하지 않습니다(총 {total_pages} 페이지). 설정을 무시하고 전체 페이지를 추출합니다 ({pdf_url}).")
                 first_page_index = -1 # 잘못된 설정이면 전체 추출하도록 변경

        # 텍스트 추출기 인스턴스 생성 (수정된 클래스 사용 - 인자 없음)
        extractor = PDFTextExtractor()

        all_text_parts = []

        # 모든 페이지 순회 (0-based index 사용)
        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            page_text = ""

            if page_idx == first_page_index:
                # 설정된 첫 페이지만 특정 영역/키워드 적용
                page_text = extractor.extract_text_from_page(
                    page=page,
                    rect_coordinates=coordinates,
                    stop_keywords=stop_keywords
                )
            else:
                # 나머지 페이지 또는 설정이 없는 경우 전체 텍스트 추출
                full_page_rect = page.rect
                page_text = extractor.extract_text_from_page(
                    page=page,
                    rect_coordinates=(full_page_rect.x0, full_page_rect.y0, full_page_rect.x1, full_page_rect.y1),
                    stop_keywords=[] # 나머지 페이지는 stop keyword 미적용
                )

            all_text_parts.append(page_text)

        return "\n".join(filter(None, all_text_parts)).strip() # 빈 문자열 제외하고 결합

    except requests.exceptions.RequestException as e:
        #logger.error(f"PDF 다운로드 중 오류 발생 ({pdf_url}): {e}")
        return ""
    # --- 제거된 부분: except fitz.fitz.FileNotFoundError: ---
    except Exception as e: # 그 외 예측 못한 오류 처리
        #logger.error(f"PDF 처리 중 예기치 못한 오류 발생 ({pdf_url}): {e}")
        # import traceback # 필요시 상세 traceback 출력
        # logger.error(traceback.format_exc())
        return ""
    finally:
        # 리소스 해제
        if doc:
            doc.close()
            #logger.debug(f"fitz Document 닫힘 ({pdf_url})")
        if pdf_buffer:
            pdf_buffer.close()
            #logger.debug(f"PDF Buffer 닫힘 ({pdf_url})")
