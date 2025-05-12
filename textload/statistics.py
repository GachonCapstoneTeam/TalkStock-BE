import pandas as pd

def count_reports_by_industry(df, industry_col='업종'):
    if industry_col not in df.columns:
        raise ValueError(f"'{industry_col}' 컬럼이 데이터프레임에 존재하지 않습니다.")

    industry_counts = df[industry_col].value_counts().reset_index()
    industry_counts.columns = ['업종', '리포트 수']
    return industry_counts