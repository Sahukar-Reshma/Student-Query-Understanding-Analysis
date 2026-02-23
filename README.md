# Student Query Understanding

An offline Python-based NLP tool designed to understand and analyze student queries.

The system predicts:

* Intent of the question
* Academic topic
* Difficulty level
* Generates a student-friendly explanation

The entire system works locally using pre-trained embeddings and does not require internet connectivity.

---

## Project Objective

* Automatically analyze student questions
* Classify intent (Explanation, Example, Doubt Clarification, Revision)
* Identify academic topic
* Determine difficulty level
* Generate structured, student-friendly answers
* Enable intelligent offline academic assistance

---

## Core Concept

This system uses **semantic understanding instead of keyword matching**.

When a student enters a query:

1. The query is converted into sentence embeddings
2. The embedding is mapped into a semantic vector space
3. Cosine similarity is used to classify topic
4. Intent and difficulty level are predicted
5. A structured JSON response is generated

The model uses lightweight transformer-based embeddings for contextual understanding.

---

## Features

* Intent classification:

  * Explanation
  * Example
  * Doubt Clarification
  * Revision

* Topic classification:

  * Backpropagation
  * Gradient Descent
  * Neural Networks
  * Optimization
  * Linear Regression

* Difficulty classification:

  * Beginner
  * Intermediate
  * Advanced

* Generates student-friendly answers

* Fully offline execution

* Modular OOPS-based design

---

## Tech Stack

**Language:** Python 3.10.0
**IDE:** VS Code

### Libraries Used:

* `numpy`
* `sentence-transformers`
* `scikit-learn`

---

## Setup / Installation

### Prerequisites

* Python 3.10+ installed
  [https://www.python.org/downloads/](https://www.python.org/downloads/)

* Git installed
  [https://git-scm.com/downloads](https://git-scm.com/downloads)

* VS Code (optional but recommended)
  [https://code.visualstudio.com/](https://code.visualstudio.com/)

---

## Steps

### Clone the Repository

```bash
git clone https://github.com/Sahukar-Reshma/Student-Query-Understanding-Analysis
cd Student-Query-Understanding-Analysis
```

---

### Install Dependencies

```bash
pip install numpy sentence-transformers scikit-learn
```

---

### Run the Project

```bash
python main.py
```

---

## Usage

* Enter a student question when prompted.
* The system analyzes the query.
* Type `exit` or `quit` to stop the program.

---

## Example Output

### Student Input

```
Student Question: I'm stuck on the gradient descent topic
```

---

### Q1: Query Analysis Output

```json
{
    "intent": "Revision",
    "topic": "Gradient Descent",
    "difficulty_level": "Intermediate",
    "answer": "Revision on Gradient Descent: This is a clear, concise explanation suitable for a student asking 'I'm stuck on the gradient descent topic'."
}
```

---

## Output Explanation

* **intent** → Type of request detected (Revision in this case)
* **topic** → Classified academic topic
* **difficulty_level** → Estimated student level
* **answer** → Generated student-friendly response

In this example, the phrase “I'm stuck” signals revision intent.
The system detects "Gradient Descent" as topic and assigns intermediate difficulty.

---

## Project Structure

```
student-query-understanding/
│
├─ main.py                      # Entry point of the project
├─ src/
│   └─ query_understanding.py   # Core module for query analysis
└─ README.md                    # Project documentation
```

---

## OOPS Design

The project follows a modular object-oriented architecture:

* Constructor initializes embedding model
* Classification logic encapsulated inside class
* Public method exposes query analysis functionality
* Internal processing abstracted from user interface

This ensures:

* Encapsulation
* Abstraction
* Modularity
* Scalability

---

##  Future Enhancements

* Fine-tuned domain-specific transformer model
* Larger dataset for better classification accuracy
* Integration with Adaptive Learning Path system
* REST API deployment
* Conversational AI extension

---

##  Repository Link

```bash
git clone https://github.com/Sahukar-Reshma/Student-Query-Understanding-Analysis
cd Student-Query-Understanding-Analysis
```
