import re
import openai
import streamlit as st
import fitz  # PyMuPDF
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from io import BytesIO
from sqlalchemy.orm import declarative_base

# PostgreSQL 데이터베이스 URL 설정
SQLALCHEMY_DATABASE_URL = 'postgresql://postgres.vqbazzxcoveyrtrhelgh:choiminseuck@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres'
api_key = st.text_input("Enter your API key:", type="password")

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

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
st.title("성취기준 평가 알려주는 프로그램")
st.write("성취기준 평가 알려주는 프로그램")

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

# 문장 선택 체크박스 추가
selected_sentences = []
st.write("해당 학년과 과목에 해당하는 문장을 선택하세요:")
for sentence in sentence_options:
    if st.checkbox(sentence):
        selected_sentences.append(sentence)

st.write("엑셀 파일을 업로드하고, 이름 열을 확인하세요.")
uploaded_excel = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

def openate(file):
    field_names = file.columns.tolist()[1:]
    name = file["이름"]
    evaluations = []
    for s in range(len(name)):
        sent = "당신은 교사입니다. 다음 학생은 "
        for i in range(1,len(field_names)+1):
            print(file.iloc[s, i].strip())
            if file.iloc[s, i].strip() == "상":
                sent += "(" + field_names[i-1].replace("\n", "") + ")의 평가에 있어서는 놀라운 성취를 보여주었습니다."
            elif file.iloc[s, i].strip() == "중":
                sent += "(" + field_names[i-1].replace("\n", "") + ")의 평가에 있어서는 좋은 모습을 보여주었습니다."
            elif file.iloc[s, i].strip() == "하":
                sent += "(" + field_names[i-1].replace("\n", "") + ")의 평가에 있어서는 부족한 모습을 보여주었습니다."
            else:
                print("@@",file.iloc[s, i].strip())
                sent += "(" + field_names[i-1].replace("\n", "") + ")의 평가에 있어서는 나쁘지 않은 수준의 모습을 보여주었습니다."
        #
        sent += "학생을 평가하는 말을 한국말로 만들어주세요"
        openai.api_key = api_key


        print(sent)
    #
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": sent + "위 학생에 대해 평가하는 말을 해주세요." + "종결어미는 -함, -보여줌 과같은 말, 주어를 생략해주세요. 4문장으로 말해주세요"},
                {"role": "user", "content": sent + "위 학생에 대해 평가하는 말을 해주세요." + "-함, -보여줌 과같은 말로 끝내고, 주어를 생략함. 평가내용을 다양한 구체적인 예시를 들어 평가해주세요. -함, -보여줌 과같은 말로 종결함. 4문장으로 말해주세요"},
            ]
        )
        evaluation = response['choices'][0]['message']['content']

        # 리스트에 평가 내용 추가
        evaluations.append(evaluation)

    file["평가"] = evaluations
    return file

if uploaded_excel is not None:
    # 엑셀 파일 읽기
    df = pd.read_excel(uploaded_excel)

    # '이름' 열이 존재하는지 확인
    if '이름' in df.columns:
        st.write("업로드된 엑셀 파일의 '이름' 열:")

        for i in range(len(selected_sentences)):
            df[selected_sentences] = ""

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True)  # 모든 열을 편집 가능하도록 설정
        grid_options = gb.build()

        # AgGrid 사용하여 데이터프레임을 인터랙티브하게 표시
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            allow_unsafe_jscode=True,
            editable=True,
        )

        # 업데이트된 데이터 가져오기
        updated_df = grid_response['data']

        # 업데이트된 데이터프레임 출력
        st.write("업데이트된 데이터프레임:")
        st.dataframe(updated_df)

        if st.button("평가결과_다운준비"):
            # 버튼 클릭 시, 그리드에서 업데이트된 데이터 가져오기
            updated_df = grid_response['data']
            result_df = openate(updated_df)

            # 엑셀 파일로 저장
            df_xlsx = to_excel(result_df)

            # 다운로드 버튼 제공
            st.download_button(
                label="평가 결과 엑셀 파일 다운로드",
                data=df_xlsx,
                file_name="평가결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 결과 표시
    else:
        st.error("'이름' 열이 엑셀 파일에 존재하지 않습니다.")


SQLALCHEMY_DATABASE_URL = 'postgresql://postgres.vqbazzxcoveyrtrhelgh:choiminseuck@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres'

# 데이터베이스 엔진 생성
engine = create_engine(SQLALCHEMY_DATABASE_URL)

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

# 데이터베이스 세션 관리를 위한 함수
@st.cache_resource
def get_db():
    return SessionLocal()
# Streamlit 앱 시작


    # 파일 업로더
uploaded_file = st.file_uploader("성취기준 관련 PDF 파일을 업로드하세요. 데이터가 자동 저장됩니다.", type="pdf")

if uploaded_file is not None:
        # PDF 파일 처리
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")

        # 정규표현식 패턴 정의
    pattern = r'\[\d+[가-힣]+\d+-\d+\]'
    pattern_number = r'\d+'
    pattern_hangeul = r'[가-힣]+'

        # 데이터베이스 세션 시작
    db = get_db()
    try:

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
                    # PostgreSQL 데이터베이스에 삽입
                    db.add(Criteria(Grade=r, Subject=s, sentence=q, patterned=p))

        db.commit()
        st.success("데이터가 PostgreSQL 데이터베이스에 저장되었습니다.")
    except Exception as e:
        db.rollback()  # 오류 발생 시 롤백
        st.error(f"데이터베이스에 데이터를 저장하는 중 오류가 발생했습니다: {e}")

        try:
            db.rollback()
        except Exception:
            st.error("롤백 중 오류가 발생했습니다.")
    finally:
        db.close()  # 세션 종료























