# 베이스 이미지 선택
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt requirements.txt

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# secrets.json 복사
COPY secrets.json /app/secrets.json

RUN ls -l /app/secrets.json

# Django 프로젝트 복사
COPY . .

# 포트 설정
EXPOSE 8000

# 실행 명령어
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
