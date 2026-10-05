import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama


# =====================================================
# CONFIGURATION
# =====================================================

OLLAMA_MODEL = "llama3.2"


# =====================================================
# STREAMLIT PAGE
# =====================================================

st.set_page_config(
    page_title="AI Career Course Recommendation System",
    page_icon="🧭",
    layout="wide"
)

# =====================================================
# LIGHT ELEGANT THEME
# =====================================================

st.markdown("""
<style>

    /* ==========================================
       MAIN APP BACKGROUND
       ========================================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8f9ff,
            #eef5ff,
            #fdf6f9
        );
        background-attachment: fixed;
    }


    /* ==========================================
       MAIN CONTENT CARD
       ========================================== */

    .main .block-container {
        background: rgba(255, 255, 255, 0.96);

        padding: 2.5rem 3rem;

        border-radius: 24px;

        margin-top: 25px;
        margin-bottom: 25px;

        box-shadow:
            0px 8px 30px rgba(80, 90, 120, 0.12);
    }


    /* ==========================================
       MAIN TITLE
       ========================================== */

    h1 {
        color: #3f4a6b !important;

        text-align: center;

        font-weight: 800;

        font-size: 36px !important;

        letter-spacing: 0.5px;
    }


    /* ==========================================
       SUB HEADINGS
       ========================================== */

    h2, h3 {
        color: #596780 !important;

        font-weight: 700;
    }


    /* ==========================================
       NORMAL TEXT
       ========================================== */

    p {
        color: #596174;

        font-size: 16px;
    }


    /* ==========================================
       TEXT INPUT
       ========================================== */

    .stTextInput input {

        background-color: #fbfcff !important;

        color: #3f4657 !important;

        border: 1.5px solid #cbd5e8 !important;

        border-radius: 12px !important;

        padding: 12px !important;

        transition: all 0.2s ease;
    }


    /* INPUT FOCUS */

    .stTextInput input:focus {

        border-color: #9aaedb !important;

        box-shadow:
            0px 0px 8px rgba(154, 174, 219, 0.25);
    }


    /* ==========================================
       TEXT AREA
       ========================================== */

    .stTextArea textarea {

        background-color: #fbfcff !important;

        color: #3f4657 !important;

        border: 1.5px solid #cbd5e8 !important;

        border-radius: 12px !important;

        padding: 12px !important;

        transition: all 0.2s ease;
    }


    /* TEXT AREA FOCUS */

    .stTextArea textarea:focus {

        border-color: #9aaedb !important;

        box-shadow:
            0px 0px 8px rgba(154, 174, 219, 0.25);
    }


    /* ==========================================
       INPUT LABELS
       ========================================== */

    .stTextInput label,
    .stTextArea label {

        color: #4d5870 !important;

        font-weight: 650 !important;

        font-size: 15px !important;
    }


    /* ==========================================
       RECOMMEND BUTTON
       ========================================== */

    .stButton > button {

        background: linear-gradient(
            90deg,
            #a8b9e8,
            #b9a9d8
        );

        color: #ffffff;

        font-size: 17px;

        font-weight: 700;

        border: none;

        border-radius: 12px;

        padding: 12px;

        transition: all 0.3s ease;
    }


    /* BUTTON HOVER */

    .stButton > button:hover {

        background: linear-gradient(
            90deg,
            #91a6dc,
            #a997ca
        );

        transform: translateY(-2px);

        box-shadow:
            0px 7px 18px rgba(120, 130, 180, 0.25);
    }


    /* ==========================================
       DATAFRAME
       ========================================== */

    [data-testid="stDataFrame"] {

        border-radius: 14px;

        overflow: hidden;

        box-shadow:
            0px 6px 18px rgba(80, 90, 120, 0.10);
    }


    /* ==========================================
       DOWNLOAD BUTTON
       ========================================== */

    .stDownloadButton > button {

        background: #a8c8c0;

        color: #ffffff;

        font-weight: 700;

        border: none;

        border-radius: 12px;

        padding: 11px;

        transition: all 0.3s ease;
    }


    /* DOWNLOAD HOVER */

    .stDownloadButton > button:hover {

        background: #8eb5ab;

        box-shadow:
            0px 6px 16px rgba(100, 140, 130, 0.20);
    }


    /* ==========================================
       ALERT / WARNING
       ========================================== */

    [data-testid="stAlert"] {

        border-radius: 12px;
    }


    /* ==========================================
       SPINNER
       ========================================== */

    .stSpinner > div {

        color: #899bc8 !important;
    }

</style>
""", unsafe_allow_html=True)

# =====================================================
# SENTENCE TRANSFORMER
# =====================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# =====================================================
# CHROMADB
# =====================================================

@st.cache_resource
def load_chromadb():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    # New collection name to avoid old database errors
    collection = client.get_or_create_collection(
        name="career_courses_v2"
    )

    courses = [

        {
            "course": "Python Programming",
            "category": "Python",
            "websites": "Coursera, Kaggle, freeCodeCamp",
            "duration": "4 weeks",
            "description": "Python programming and problem solving."
        },

        {
            "course": "SQL for Data Analysis",
            "category": "SQL",
            "websites": "Coursera, Kaggle, W3Schools",
            "duration": "3 weeks",
            "description": "SQL queries, joins and database analysis."
        },

        {
            "course": "Statistics for Data Science",
            "category": "Statistics",
            "websites": "Khan Academy, Coursera, edX",
            "duration": "5 weeks",
            "description": "Probability, distributions, hypothesis testing and regression."
        },

        {
            "course": "Machine Learning",
            "category": "Machine Learning",
            "websites": "Coursera, Kaggle, edX",
            "duration": "8 weeks",
            "description": "Regression, classification, clustering and machine learning."
        },

        {
            "course": "Deep Learning",
            "category": "Deep Learning",
            "websites": "Coursera, Kaggle, edX",
            "duration": "8 weeks",
            "description": "Neural networks, CNNs and deep learning."
        },

        {
            "course": "Data Visualization",
            "category": "Data Visualization",
            "websites": "Kaggle, Coursera, freeCodeCamp",
            "duration": "3 weeks",
            "description": "Charts, dashboards and data visualization."
        },

        {
            "course": "Pandas for Data Analysis",
            "category": "Pandas",
            "websites": "Kaggle, DataCamp, freeCodeCamp",
            "duration": "3 weeks",
            "description": "Data cleaning and analysis using Pandas."
        },

        {
            "course": "NumPy Fundamentals",
            "category": "NumPy",
            "websites": "Kaggle, DataCamp, W3Schools",
            "duration": "2 weeks",
            "description": "Numerical computing and arrays."
        },

        {
            "course": "Git and GitHub",
            "category": "Git and GitHub",
            "websites": "GitHub Skills, Coursera, freeCodeCamp",
            "duration": "2 weeks",
            "description": "Version control and GitHub."
        },

        {
            "course": "Data Structures and Algorithms",
            "category": "Data Structures",
            "websites": "GeeksforGeeks, Coursera, HackerRank",
            "duration": "8 weeks",
            "description": "Arrays, linked lists, trees, graphs and algorithms."
        },

        {
            "course": "Full Stack Web Development",
            "category": "Web Development",
            "websites": "freeCodeCamp, Coursera, Udemy",
            "duration": "10 weeks",
            "description": "Frontend, backend and APIs."
        },

        {
            "course": "Cloud Computing Fundamentals",
            "category": "Cloud Computing",
            "websites": "AWS, Coursera, Microsoft Learn",
            "duration": "5 weeks",
            "description": "Cloud services and deployment."
        },

        {
            "course": "Natural Language Processing",
            "category": "NLP",
            "websites": "Coursera, Kaggle, edX",
            "duration": "7 weeks",
            "description": "Text processing, embeddings and NLP."
        },

        {
            "course": "Computer Vision",
            "category": "Computer Vision",
            "websites": "Coursera, Kaggle, edX",
            "duration": "7 weeks",
            "description": "Image processing and computer vision."
        },

        {
            "course": "Power BI Data Analysis",
            "category": "Power BI",
            "websites": "Microsoft Learn, Coursera, Udemy",
            "duration": "4 weeks",
            "description": "Dashboards, reports and business intelligence."
        },

        {
            "course": "Data Science Fundamentals",
            "category": "Data Science",
            "websites": "Coursera, Kaggle, edX",
            "duration": "6 weeks",
            "description": "Data analysis, statistics and machine learning."
        },

        {
            "course": "Artificial Intelligence Fundamentals",
            "category": "Artificial Intelligence",
            "websites": "Coursera, edX, Udacity",
            "duration": "6 weeks",
            "description": "Artificial intelligence and machine learning concepts."
        },

        {
            "course": "Generative AI",
            "category": "Generative AI",
            "websites": "Coursera, DeepLearning.AI, Microsoft Learn",
            "duration": "5 weeks",
            "description": "LLMs, prompt engineering and generative AI."
        }
    ]


    # =================================================
    # ADD DATA TO CHROMADB
    # =================================================

    if collection.count() == 0:

        documents = []
        metadatas = []
        ids = []

        for index, course in enumerate(courses):

            document = (
                course["course"]
                + " "
                + course["category"]
                + " "
                + course["description"]
            )

            documents.append(document)
            metadatas.append(course)
            ids.append(str(index))

        embeddings = embedding_model.encode(
            documents
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    return collection


collection = load_chromadb()


# =====================================================
# OLLAMA
# =====================================================

def ask_ollama(prompt):

    try:

        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt
        )

        return response["response"]

    except Exception as error:

        return "ERROR: " + str(error)


# =====================================================
# SEARCH COURSES
# =====================================================

def search_courses(query):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10
    )

    courses = []

    if results["metadatas"]:

        for metadata_group in results["metadatas"]:

            for metadata in metadata_group:

                # Safe access to prevent KeyError
                courses.append({
                    "course": metadata.get(
                        "course",
                        "Unknown Course"
                    ),

                    "category": metadata.get(
                        "category",
                        "General"
                    ),

                    "websites": metadata.get(
                        "websites",
                        "Coursera, Udemy, YouTube"
                    ),

                    "duration": metadata.get(
                        "duration",
                        "Not specified"
                    ),

                    "description": metadata.get(
                        "description",
                        ""
                    )
                })

    return courses


# =====================================================
# COURSE RECOMMENDATION
# =====================================================

def recommend_courses(
    career,
    completed_courses,
    available_time,
    resources
):

    resource_text = ""

    for resource in resources:

        resource_text += f"""
Course: {resource["course"]}
Category: {resource["category"]}
Websites: {resource["websites"]}
Duration: {resource["duration"]}
Description: {resource["description"]}

-------------------------
"""


    prompt = f"""
You are an expert AI career course recommendation system.

TARGET CAREER:
{career}

COURSES ALREADY COMPLETED:
{completed_courses}

AVAILABLE LEARNING TIME:
{available_time}

COURSE INFORMATION FROM DATABASE:
{resource_text}


TASK:

Recommend the courses the user should learn to become
job-ready for the target career.


IMPORTANT RULES:

1. Recommend COURSES only.

2. Do NOT create a skills-required section.

3. Do NOT display a list of required skills.

4. Do NOT recommend courses that duplicate the
   user's completed courses.

5. Recommend exactly 8 to 10 different courses.

6. Consider the user's available learning time.

7. You can use the database information as supporting
   information.

8. You can also recommend real courses that are not
   present in the database.

9. Recommend useful and well-known learning platforms.

10. Give 2 or 3 websites for every course.

11. Do not include:
   - Reason
   - Difficulty
   - Importance
   - Project
   - Required skills

12. Do not repeat courses.

13. Arrange courses in a sensible learning order.

14. Required Depth must be one of:
   Basic
   Intermediate
   Advanced
   Practical


USE EXACTLY THIS FORMAT:


Course 1:
Course Name: ...
Required Depth: Basic/Intermediate/Advanced/Practical
Duration: ...
Websites:
1. ...
2. ...
3. ...


Course 2:
Course Name: ...
Required Depth: Basic/Intermediate/Advanced/Practical
Duration: ...
Websites:
1. ...
2. ...
3. ...


Continue until you have 8 to 10 courses.
"""


    return ask_ollama(prompt)


# =====================================================
# CONVERT OLLAMA OUTPUT TO DATAFRAME
# =====================================================

def recommendations_to_dataframe(recommendations):

    import re
    import pandas as pd

    courses = []

    # Split response into Course 1, Course 2, etc.
    blocks = re.split(
        r"Course\s+\d+\s*:",
        recommendations,
        flags=re.IGNORECASE
    )

    for block in blocks:

        if not block.strip():
            continue

        course_name = re.search(
            r"Course Name:\s*(.*)",
            block,
            re.IGNORECASE
        )

        depth = re.search(
            r"Required Depth:\s*(.*)",
            block,
            re.IGNORECASE
        )

        duration = re.search(
            r"Duration:\s*(.*)",
            block,
            re.IGNORECASE
        )

        # Get all website lines
        websites = re.findall(
            r"\d+\.\s*(.*)",
            block
        )

        # Combine all websites into one cell
        websites_text = ", ".join(
            website.strip()
            for website in websites
        )

        courses.append({
            "Course Name":
                course_name.group(1).strip()
                if course_name else "Not specified",

            "Required Depth":
                depth.group(1).strip()
                if depth else "Not specified",

            "Duration":
                duration.group(1).strip()
                if duration else "Not specified",

            "Websites":
                websites_text
        })

    return pd.DataFrame(courses)

# =====================================================
# USER INTERFACE
# =====================================================

st.title(
    "🗺️ AI-Powered Career Course Recommendation System"
)


st.write(
    "Enter your career goal, completed courses and "
    "available learning time to receive personalized "
    "course recommendations."
)


# =====================================================
# INPUT 1
# =====================================================

career = st.text_input(
    "🎯 Target Career / Job",
    placeholder="Example: Data Scientist"
)


# =====================================================
# INPUT 2
# =====================================================

completed_courses = st.text_area(
    "📚 Courses Already Completed",
    placeholder="Example: Python Programming, SQL Basics"
)


# =====================================================
# INPUT 3
# =====================================================

available_time = st.text_input(
    "⏳ Available Learning Time",
    placeholder="Example: 1 year"
)


# =====================================================
# RECOMMEND BUTTON
# =====================================================

if st.button(
    "🚀 Recommend Courses",
    use_container_width=True
):

    if not career:

        st.warning(
            "Please enter your target career."
        )

        st.stop()


    if not available_time:

        st.warning(
            "Please enter your available learning time."
        )

        st.stop()


    # =================================================
    # SEARCH QUERY
    # =================================================

    search_query = (
        career
        + " "
        + completed_courses
    )


    # =================================================
    # CHROMADB SEARCH
    # =================================================

    with st.spinner(
        "🔎 Finding relevant course information..."
    ):

        resources = search_courses(
            search_query
        )


    # =================================================
    # OLLAMA RECOMMENDATION
    # =================================================

    with st.spinner(
        "😂 Identifying your course recommendations..."
    ):

        recommendations = recommend_courses(
            career,
            completed_courses,
            available_time,
            resources
        )


    # =================================================
    # ERROR CHECK
    # =================================================

    if recommendations.startswith("ERROR"):

        st.error(
            recommendations
        )

        st.stop()


    # =================================================
    # DISPLAY RESULTS AS DATAFRAME
    # =================================================

    st.subheader(
        "📚 Recommended Courses"
    )

    recommendation_df = recommendations_to_dataframe(
        recommendations
    )

    if not recommendation_df.empty:

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Unable to convert the recommendations into a table."
        )


    # =================================================
    # DOWNLOAD
    # =================================================

    download_text = (
        "AI CAREER COURSE RECOMMENDATIONS\n"
        "================================\n\n"
    )

    download_text += (
        f"Target Career: {career}\n"
    )

    download_text += (
        f"Available Learning Time: "
        f"{available_time}\n\n"
    )

    download_text += recommendations


    st.download_button(
        label="📥 Download Recommended Courses",
        data=download_text,
        file_name="AI_Career_Courses_Recommendations.txt",
        mime="text/plain",
        use_container_width=True
    )