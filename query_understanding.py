import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier

class StudentQueryUnderstanding:
    """
    Offline student query understanding:
    - Intent classification (Explanation, Example, Doubt Clarification, Revision)
    - Topic classification (Backpropagation, Gradient Descent, Neural Networks, etc.)
    - Difficulty classification (Beginner, Intermediate, Advanced)
    - Generates student-friendly answer
    """

    def __init__(self):
        # Load MiniLM embeddings
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # ---------- INTENT ----------
        self.intents = ["Explanation", "Example", "Doubt Clarification", "Revision"]

        # Expanded training examples for robust intent classification
        self.intent_sentences = [
            # Explanation
            "I don't understand this concept",
            "Can you explain this to me?",
            "I am confused about this topic",
            "I can't grasp this idea",
            "Please explain this concept",

            # Example
            "Can you give me an example?",
            "Show me an example problem",
            "I want to see a worked example",
            "Give me an example for practice",
            "Demonstrate with an example",

            # Doubt Clarification
            "I have a doubt about this topic",
            "I am stuck on this",
            "I don't get this part",
            "I'm confused about this step",
            "Can you clarify this question?",

            # Revision
            "Please help me revise this",
            "I want to review this topic",
            "Help me go over this again",
            "I need a quick revision",
            "Can we revise this topic?"
        ]

        self.intent_labels = [
            0,0,0,0,0,  # Explanation
            1,1,1,1,1,  # Example
            2,2,2,2,2,  # Doubt Clarification
            3,3,3,3,3   # Revision
        ]

        intent_embeddings = self.embed_model.encode(self.intent_sentences)
        self.intent_classifier = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        self.intent_classifier.fit(intent_embeddings, self.intent_labels)

        # ---------- TOPIC ----------
        self.topics = ["Backpropagation", "Gradient Descent", "Neural Networks", "Optimization", "Linear Regression"]
        topic_sentences = [
            "I don't understand backpropagation",
            "Explain gradient descent",
            "Tell me about neural networks",
            "I have a question on optimization",
            "Explain linear regression"
        ]
        topic_embeddings = self.embed_model.encode(topic_sentences)
        self.topic_classifier = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        self.topic_classifier.fit(topic_embeddings, self.topics)

    # ---------- ANALYZE QUERY ----------
    def analyze_query(self, query: str):
        # Embed query
        query_emb = self.embed_model.encode([query])

        # Intent prediction
        intent_idx = self.intent_classifier.predict(query_emb)[0]
        intent = self.intents[intent_idx]

        # Topic prediction
        topic = self.topic_classifier.predict(query_emb)[0]

        # Difficulty heuristic
        query_lower = query.lower()
        if any(word in query_lower for word in ["simple", "beginner", "basic", "easy"]):
            difficulty = "Beginner"
        elif any(word in query_lower for word in ["hard", "complex", "advanced", "difficult"]):
            difficulty = "Advanced"
        else:
            difficulty = "Intermediate"

        # Student-friendly answer template
        answer = f"{intent} on {topic}: This is a clear, concise explanation suitable for a student asking '{query}'."

        # Return JSON-ready dictionary
        return {
            "intent": intent,
            "topic": topic,
            "difficulty_level": difficulty,
            "answer": answer
        }
