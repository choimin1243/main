import re
import streamlit as st
import fitz  # PyMuPDF
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from io import BytesIO

# Import both OpenAI and Google Generative AI
import openai
import google.generativeai as genai

# PostgreSQL database URL setup
SQLALCHEMY_DATABASE_URL = 'postgresql://postgres.vqbazzxcoveyrtrhelgh:choiminseuck@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres'
api_key = st.text_input("Enter your GPT or Gemini API key:", type="password")
api_type = st.selectbox("Select API type:", ["GPT", "Gemini"])

# Initialize the selected API
if api_type == "GPT":
    openai.api_key = api_key
else:
    genai.configure(api_key=api_key)


def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


# Create database engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)

# Session local setup
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base model class
Base = declarative_base()


# Define table
class Criteria(Base):
    __tablename__ = 'criteria'
    id = Column(Integer, primary_key=True, index=True)
    Grade = Column(Integer, nullable=False)
    Subject = Column(String, nullable=False)
    sentence = Column(String, nullable=False)
    patterned = Column(String, nullable=False)


# Create table (can be skipped if it already exists)
Base.metadata.create_all(bind=engine)

# Streamlit page setup
st.title("성취기준 평가 알려주는 프로그램")
st.write("성취기준 평가 알려주는 프로그램")

# Start database session
db = SessionLocal()

# Get saved grade values
grades = db.query(Criteria.Grade).distinct().all()
grade_options = [str(grade[0]) for grade in grades]

# Add grade selection dropdown
selected_grade = st.selectbox("저장된 학년을 선택하세요", grade_options)

# Get subject list based on selected grade
subjects = db.query(Criteria.Subject).filter(Criteria.Grade == selected_grade).distinct().all()
subject_options = [subject[0] for subject in subjects]

# Add subject selection dropdown
selected_subject = st.selectbox("해당 학년에 해당하는 과목을 선택하세요", subject_options)

# Get sentence list based on selected grade and subject
sentences = db.query(Criteria.sentence).filter(Criteria.Grade == selected_grade,
                                               Criteria.Subject == selected_subject).distinct().all()
sentence_options = [sentence[0] for sentence in sentences]

# Add sentence selection checkbox
selected_sentences = []
st.write("해당 학년과 과목에 해당하는 문장을 선택하세요:")
for sentence in sentence_options:
    if st.checkbox(sentence):
        selected_sentences.append(sentence)

st.write("엑셀 파일을 업로드하고, 이름 열을 확인하세요.")
uploaded_excel = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx", "xls"])


def generate_evaluation(sent):
    prompt = f"{sent} 위 학생에 대해 평가하는 말을 해주세요. 종결어미는 -함, -보여줌 과같은 말, 주어를 생략해주세요. 평가내용을 다양한 구체적인 예시를 들어 평가해주세요."

    if api_type == "GPT":
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 교사입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    else:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt+"-함. -하였음과 같이 종결어미를 써주시고 4문장으로 완성해주세요. 주어를 생략해주세요. 예를들어 '학생은 ~ 수행함'과같이 여러문장을 적어주세요")
        return response.text


def openate(file):
    field_names = file.columns.tolist()[1:]
    name = file["이름"]
    evaluations = []
    for s in range(len(name)):
        sent = "당신은 교사입니다. 다음 학생은 "
        for i in range(1, len(field_names) + 1):
            if file.iloc[s, i].strip() == "상":
                sent += f"({field_names[i - 1].replace('/n', '')})의 평가에 있어서는 놀라운 성취를 보여주었습니다."
            elif file.iloc[s, i].strip() == "중":
                sent += f"({field_names[i - 1].replace('/n', '')})의 평가에 있어서는 좋은 모습을 보여주었습니다."
            elif file.iloc[s, i].strip() == "하":
                sent += f"({field_names[i - 1].replace('/n', '')})의 평가에 있어서는 부족한 모습을 보여주었습니다."
            else:
                sent += f"({field_names[i - 1].replace('/n', '')})의 평가에 있어서는 나쁘지 않은 수준의 모습을 보여주었습니다."

        sent += "학생을 평가하는 말을 한국말로 만들어주세요"

        evaluation = generate_evaluation(sent)
        evaluations.append(evaluation)

    file["평가"] = evaluations
    return file


if uploaded_excel is not None:
    # Read Excel file
    df = pd.read_excel(uploaded_excel)

    # Check if '이름' column exists
    if '이름' in df.columns:
        st.write("업로드된 엑셀 파일의 '이름' 열:")

        for i in range(len(selected_sentences)):
            df[selected_sentences] = ""

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True)  # Set all columns to be editable
        grid_options = gb.build()

        # Use AgGrid to display the dataframe interactively
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            allow_unsafe_jscode=True,
            editable=True,
        )

        # Get updated data
        updated_df = grid_response['data']

        # Display updated dataframe
        st.write("업데이트된 데이터프레임:")
        st.dataframe(updated_df)

        if st.button("평가결과_다운준비"):
            # When button is clicked, get updated data from the grid
            updated_df = grid_response['data']
            result_df = openate(updated_df)

            # Save as Excel file
            df_xlsx = to_excel(result_df)

            # Provide download button
            st.download_button(
                label="평가 결과 엑셀 파일 다운로드",
                data=df_xlsx,
                file_name="평가결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.error("'이름' 열이 엑셀 파일에 존재하지 않습니다.")


# Database session management function
@st.cache_resource
def get_db():
    return SessionLocal()


# File uploader
uploaded_file = st.file_uploader("성취기준 관련 PDF 파일을 업로드하세요. 데이터가 자동 저장됩니다.", type="pdf")

if uploaded_file is not None:
    # Process PDF file
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    # Define regex patterns
    pattern = r'\[\d+[가-힣]+\d+-\d+\]'
    pattern_number = r'\d+'
    pattern_hangeul = r'[가-힣]+'

    # Start database session
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
                    # Insert into PostgreSQL database
                    db.add(Criteria(Grade=r, Subject=s, sentence=q, patterned=p))

        db.commit()
        st.success("데이터가 PostgreSQL 데이터베이스에 저장되었습니다.")
    except Exception as e:
        db.rollback()  # Rollback on error
        st.error(f"데이터베이스에 데이터를 저장하는 중 오류가 발생했습니다: {e}")

        try:
            db.rollback()
        except Exception:
            st.error("롤백 중 오류가 발생했습니다.")
    finally:
        db.close()  # Close session
