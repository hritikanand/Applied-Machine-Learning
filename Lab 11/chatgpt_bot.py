from transformers import pipeline

# Load the QA model pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

#  Updated context passage
context = """
Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment 
in order to maximize some notion of cumulative reward. It is one of the three basic machine learning paradigms, alongside supervised 
learning and unsupervised learning. In reinforcement learning, an agent interacts with its environment, learns from the consequences 
of its actions, and adjusts its behavior to improve performance over time. Applications of reinforcement learning include robotics, 
game playing, recommendation systems, and autonomous vehicles.
"""

#  Questions to ask
questions = [
    "What is reinforcement learning?",
    "What are the three basic types of machine learning?",
    "Where is reinforcement learning applied?"
]

#  Ask all questions
print(" Local Q&A with distilBERT model\n")
for i, q in enumerate(questions, start=1):
    result = qa_pipeline(question=q, context=context)
    print(f"Q{i}: {q}")
    print(f"A{i}: {result['answer']}")
    print("-" * 60)

print(" Done! Take a screenshot of this output for your report.")
