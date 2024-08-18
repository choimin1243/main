import re
import streamlit as st
import fitz  # PyMuPDF
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# PostgreSQL 데이터베이스 URL 설정
SQLALCHEMY_DATABASE_URL = 'postgresql://postgres.vqbazzxcoveyrtrhelgh:choiminseuck@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres'

# 데이터베이스 엔진 생성
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True  # SQLAlchemy가 생성하는 SQL 쿼리를 로깅합니다 (선택사항)
)

# 세션 로컬 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 기본 모델 클래스 생성
Base = declarative_base()

# 테이블 정의
class Criteria(Base):
    __tablename__ = 'criteria'
    id = Column(Integer, primary_key=True, index=True)
    Grade = Column(Integer, nullable=False)
    Subject = Column(String, nullable=False)
    sentence = Column(String, nullable=False)
    patterned = Column(String, nullable=False)

# 테이블 생성 (이미 존재하면 생략 가능)
Base.metadata.create_all(bind=engine)

# Streamlit 페이지 설정
st.title("PDF 패턴 추출 및 PostgreSQL 저장")
st.write("PDF 파일을 업로드하고, 패턴을 찾아 PostgreSQL 데이터베이스에 저장합니다.")

# 데이터베이스 세션 시작
db = SessionLocal()

# 저장된 학년 값을 가져오기
grades = db.query(Criteria.Grade).distinct().all()
grade_options = [str(grade[0]) for grade in grades]

# 학년 선택 드롭다운 추가
selected_grade = st.selectbox("저장된 학년을 선택하세요", grade_options)

# 선택된 학년에 따른 과목(Subject) 목록 가져오기
subjects = db.query(Criteria.Subject).filter(Criteria.Grade == selected_grade).distinct().all()
subject_options = [subject[0] for subject in subjects]

# 과목 선택 드롭다운 추가
selected_subject = st.selectbox("해당 학년에 해당하는 과목을 선택하세요", subject_options)

# 선택된 학년과 과목에 따른 문장(sentence) 목록 가져오기
sentences = db.query(Criteria.sentence).filter(Criteria.Grade == selected_grade, Criteria.Subject == selected_subject).distinct().all()
sentence_options = [sentence[0] for sentence in sentences]

# 문장 선택 드롭다운 추가
selected_sentence = st.selectbox("해당 학년과 과목에 해당하는 문장을 선택하세요", sentence_options)

# 파일 업로드 위젯
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file is not None:
    # PDF 파일 읽기
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    # 정규표현식 패턴 정의
    pattern = r'\[\d+[가-힣]+\d+-\d+\]'
    pattern_number = r'\d+'
    pattern_hangeul = r'[가-힣]+'

    # 모든 페이지에서 패턴 찾기 및 데이터베이스에 저장
    for i in range(len(pdf_document)):
        page = pdf_document[i]
        text = page.get_text()

        matches = re.findall(pattern + r'\s*(.*?다\.)', text, re.S)
        match = re.findall(pattern, text, re.S)

        list_grade = []
        list_subject = []
        if match:
            for name in match:
                if re.search(pattern_number, name):
                    list_grade.append(re.search(pattern_number, name)[0])

                if re.search(pattern_hangeul, name):
                    list_subject.append(re.search(pattern_hangeul, name)[0])

        if matches and match:
            for p, q, r, s in zip(match, matches, list_grade, list_subject):
                # PostgreSQL 데이터베이스에 삽입 (선택한 학년, 과목 및 문장과 매칭되는 경우에만)
                if r == selected_grade and s == selected_subject and q == selected_sentence:
                    db.add(Criteria(Grade=r, Subject=s, sentence=q, patterned=p))

    # 변경사항 커밋 및 연결 종료
    db.commit()
    db.close()

    st.success("데이터가 PostgreSQL 데이터베이스에 저장되었습니다.")
else:
    st.warning("PDF 파일을 업로드하세요.")
