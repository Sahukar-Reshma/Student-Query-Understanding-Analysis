from src.query_understanding import StudentQueryUnderstanding
import json

# Initialize the module
query_module = StudentQueryUnderstanding()

print("===== Q1: Student Query Understanding =====")
print("Type 'exit' to quit.\n")

while True:
    student_question = input("Student Question: ")
    if student_question.lower() in ["exit", "quit"]:
        break

    result = query_module.analyze_query(student_question)
    print("\n--- Q1: Query Analysis Output ---")
    print(json.dumps(result, indent=4))
    print("\n" + "="*80 + "\n")
