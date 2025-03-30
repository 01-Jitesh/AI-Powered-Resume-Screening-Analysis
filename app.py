import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import csv
import os

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Adjust file name/path as needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))      # Adjust file name/path as needed
le = pickle.load(open('encoder.pkl', 'rb'))       # Adjust file name/path as needed

# --------------------------
# Utility Functions
# --------------------------
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# --------------------------
# Resume Section Extraction & Analysis
# --------------------------
def extract_sections(resume_text, headings=None):
    if headings is None:
        headings = [
            "Objective", "Summary", "Profile", "Experience", "Work Experience",
            "Education", "Skills", "Projects", "Certifications", "Awards",
            "Personal Information", "Contact Information"
        ]
    pattern = r'(?P<heading>' + '|'.join(headings) + r')\s*[:\-]?\s*(?P<content>.*?)(?=(?:' + '|'.join(headings) + r')\s*[:\-]|$)'
    matches = re.finditer(pattern, resume_text, re.IGNORECASE | re.DOTALL)
    sections = {}
    for match in matches:
        heading = match.group("heading").strip().title()
        content = match.group("content").strip()
        if content:
            sections[heading] = content
    return sections

def summarize_and_score(section_name, section_text):
    sentences = re.split(r'(?<=[.!?])\s+', section_text)
    summary = " ".join(sentences[:2]) if len(sentences) >= 2 else section_text
    word_count = len(section_text.split())
    score = 0
    if section_name.lower() in ['education']:
        keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd']
        bonus = sum(1 for kw in keywords if kw in section_text.lower())
        score = word_count + bonus * 5
    elif section_name.lower() in ['experience', 'work experience']:
        dates = re.findall(r'\b(19|20)\d{2}\b', section_text)
        score = word_count + len(dates) * 10
    elif section_name.lower() == 'skills':
        skills_list = [s.strip() for s in re.split(r',|\n', section_text) if s.strip()]
        score = len(skills_list) * 10
    else:
        score = word_count
    return summary, score

def save_feedback(feedback_data, filename="feedback.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["name", "filename", "predicted_category", "feedback", "comments"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(feedback_data)

# --------------------------
# Streamlit App Layout
# --------------------------
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("Resume Category Prediction & Section Analysis App")
    st.markdown("Upload a resume in PDF, DOCX, or TXT format to get the predicted job category, a breakdown of its sections (with summaries and scores), and to provide feedback on the prediction.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"],
                                       help="Choose a resume file (PDF, DOCX, or TXT) to analyze.")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing file upload and extracting text..."):
                resume_text = handle_file_upload(uploaded_file)
            st.success("Successfully extracted text from the uploaded resume.")

            if st.checkbox("Show Extracted Text", False, help="Toggle to view the full extracted text."):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            with st.spinner("Predicting resume category..."):
                category = pred(resume_text)
            st.subheader("Predicted Category")
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

            st.subheader("Extracted Resume Sections with Summary & Score")
            with st.spinner("Extracting and analyzing resume sections..."):
                sections = extract_sections(resume_text)
            if sections:
                for heading, content in sections.items():
                    summary, score = summarize_and_score(heading, content)
                    with st.expander(heading, expanded=False):
                        st.markdown(f"**Summary:** {summary}")
                        st.markdown(f"**Score:** {score}")
                        st.markdown("---")
                        st.write(content)
            else:
                st.warning("No sections were found. Please ensure your resume contains recognizable headings.")

            st.subheader("Feedback")
            st.markdown("Was the predicted category accurate?")
            feedback_choice = st.radio("Select an option:", ("Yes", "No"), help="Select 'Yes' if the prediction is correct, otherwise 'No'.")
            user_name = st.text_input("Your Name", help="Enter your name for feedback tracking.")
            comments = st.text_area("Additional Comments (optional):", help="Provide any additional feedback or suggestions here.")
            if st.button("Submit Feedback", help="Click to submit your feedback."):
                feedback_data = {
                    "name": user_name,
                    "filename": uploaded_file.name,
                    "predicted_category": category,
                    "feedback": feedback_choice,
                    "comments": comments
                }
                save_feedback(feedback_data)
                st.success("Thank you for your feedback!")
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
