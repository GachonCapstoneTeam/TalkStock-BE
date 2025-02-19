from django.test import TestCase
from pymongo import MongoClient

from textload.views import PDFTextExtractor


class MongoDBTestCase(TestCase):
    def setUp(self):
        # MongoDB 연결 설정
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client['report_database']
        self.collection = self.db['reports']

    def tearDown(self):
        # 테스트 후 데이터 정리
        self.collection.delete_many({})

    def test_mongo_insert_and_retrieve(self):
        # 데이터 삽입
        test_data = {"Title": "Test Report", "PDF Content": "Sample content"}
        self.collection.insert_one(test_data)

        # 데이터 확인
        retrieved_data = self.collection.find_one({"Title": "Test Report"})
        self.assertIsNotNone(retrieved_data)
        self.assertEqual(retrieved_data["PDF Content"], "Sample content")


from django.test import TestCase
from io import BytesIO
import fitz  # PyMuPDF

class PDFTextExtractorTestCase(TestCase):
    def test_extract_text(self):
        # 가짜 PDF 데이터를 생성
        pdf_data = BytesIO(b"dummy_pdf_data")
        extractor = PDFTextExtractor(pdf_data)

        # 추출 테스트
        result = extractor.extract_text(
            page_number=0,
            rect_coordinates=(0, 0, 100, 100),
        )
        self.assertEqual(result, "")



from rest_framework.test import APITestCase
from rest_framework import status

class APITestCaseExample(APITestCase):
    def test_content_api(self):
        response = self.client.get("/content/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("contents", response.json())