# AI Career Course Recommendation System

📌 ## Project Overview

The AI Career Course Recommendation System is a web-based application designed to recommend suitable learning courses based on a user's target career, courses already completed, and available learning time.

The system uses AI to analyze the user's requirements and provide a structured list of recommended courses with their required depth, duration, and useful learning websites.

🎯 ## Objectives

- Recommend suitable courses for a selected target career.
- Consider the courses already completed by the user.
- Consider the user's available learning time.
- Provide 8–10 recommended courses in a clear format.
- Provide useful websites for learning each course.
- Allow users to download the recommendations as a CSV file.

📚 ## Main Features

- 🎯 Target Career input
- 📖 Completed Courses input
- ⏱️ Available Learning Time input
- 🤖 AI-based course recommendations
- 📊 DataFrame-based recommendation table
- 🌐 Multiple learning websites for each course
- 📥 CSV download option

⚙️ ## How It Works

1. The user enters their target career.
2. The user enters the courses they have already completed.
3. The user enters their available learning time.
4. Relevant course information is retrieved from the course database.
5. AI analyzes the provided information.
6. The system generates 8–10 course recommendations.
7. The recommendations are displayed in a table.
8. The user can download the results as a CSV file.

🛠️ ## Technologies Used

- Python
- Streamlit
- Ollama
- ChromaDB
- Sentence Transformers
- Pandas

📂 ## Project Structure

```text
AI-Career-Course-Recommender/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
