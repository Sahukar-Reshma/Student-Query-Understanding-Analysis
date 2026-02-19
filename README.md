# Student Query Understanding

An **offline Python tool** for understanding and analyzing student queries.  
It predicts the **intent**, **topic**, and **difficulty level** of a student question and generates a **student-friendly answer**.

---

## Features

- **Intent classification:** Explanation, Example, Doubt Clarification, Revision  
- **Topic classification:** Backpropagation, Gradient Descent, Neural Networks, Optimization, Linear Regression  
- **Difficulty classification:** Beginner, Intermediate, Advanced  
- **Generates student-friendly answers** based on query analysis  
- Fully **offline**, works locally using Python and pre-trained embeddings  

---

## Tech Stack

- **Language:** Python 3.10.0  
- **IDE:** VS Code  
- **Libraries:** 
  - `numpy`
  - `sentence-transformers`
  - `scikit-learn`

---

## Setup / Installation

### Prerequisites

- Python 3.10+ installed: [Download Python](https://www.python.org/downloads/release/python-3100/)  
- Git installed: [Download Git]([https://git-scm.com/downloads](https://desktop.github.com/download/))  
- VS Code (optional but recommended): [Download VS Code]([https://code.visualstudio.com/](https://code.visualstudio.com/download))  

### Steps

1. **Clone the repository**
```bash```

2.git clone https://github.com/Sahukar-Reshma/Student-Query-Understanding-Analysis
cd Student-Query-Understanding-Analysis

3.Install dependencies

pip install numpy sentence-transformers scikit-learn


3.Run the project

python main.py


**Usage**
Enter a student question when prompted.

Type exit or quit to stop the program.

4. Example:

Student Question: Can you explain backpropagation?
--- Output ---
{
    "intent": "Explanation",
    "topic": "Backpropagation",
    "difficulty_level": "Intermediate",
    "answer": "Explanation on Backpropagation: This is a clear, concise explanation suitable for a student asking 'Can you explain backpropagation?'."
}

**Project Structure**
student-query-understanding/
│
├─ main.py                  # Entry point of the project
├─ src/
│   └─ query_understanding.py  # Core module for query analysis
└─ README.md                # Project documentation


git clone https://github.com/your-username/student-query-understanding.git
cd student-query-understanding
