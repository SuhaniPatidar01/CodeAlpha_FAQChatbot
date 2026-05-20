from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# FAQ Questions
questions = [
    "What is your name?",
    "How are you?",
    "What is AI?",
    "What is machine learning?",
    "What is deep learning?",
    "What is NLP?",
    "What is chatbot?",
    "Who created you?",
    "What is Python?",
    "What is programming?",
    "What is coding?",
    "Can you help me?",
    "What can you do?",
    "What is data science?",
    "What is computer vision?",
    "What is technology?",
    "What is the internet?",
    "Can students learn AI?",
    "How can I start learning AI?",
    "Is Python good for beginners?",
    "Can I learn coding for free?",
    "What is the best language for AI?",
    "How can I improve coding skills?",
    "What are the benefits of AI?",
    "Can AI replace humans?",
    "Is AI dangerous?",
    "What is generative AI?",
    "What is ChatGPT?",
    "How does AI work?",
    "Can AI think like humans?",
    "What are neural networks?",
    "What is automation?",
    "What are AI tools?",
    "What is cloud computing?",
    "What is cybersecurity?",
    "What is robotics?",
    "What is an algorithm?",
    "What is a database?",
    "Can AI create images?",
    "Can AI write code?"
]

# FAQ Answers
answers = [
    "I am an AI technology chatbot.",
    "I am doing great!",
    "AI stands for Artificial Intelligence.",
    "Machine learning is a branch of AI that learns from data.",
    "Deep learning is a part of machine learning using neural networks.",
    "NLP stands for Natural Language Processing.",
    "A chatbot is a program that talks with users.",
    "I was created using Python and Streamlit.",
    "Python is a popular programming language.",
    "Programming means giving instructions to computers.",
    "Coding is the process of writing computer programs.",
    "Yes, I can answer technology and AI related questions.",
    "I can answer FAQ-based AI and technology questions.",
    "Data science is the study of data analysis and insights.",
    "Computer vision helps computers understand images and videos.",
    "Technology refers to tools and systems created using science.",
    "The internet is a global network connecting computers worldwide.",
    "Yes, students can definitely learn AI step by step.",
    "You can start AI by learning Python and machine learning basics.",
    "Yes, Python is one of the best languages for beginners.",
    "Yes, many free resources are available online for coding.",
    "Python is one of the most popular languages for AI.",
    "You can improve coding skills through practice and projects.",
    "AI helps automate tasks and solve complex problems.",
    "AI may automate some jobs but humans are still important.",
    "AI can be harmful if used irresponsibly.",
    "Generative AI creates text, images, music and more.",
    "ChatGPT is an AI chatbot developed by OpenAI.",
    "AI works using data, algorithms and machine learning models.",
    "AI can simulate thinking but does not think exactly like humans.",
    "Neural networks are systems inspired by the human brain.",
    "Automation means using technology to perform tasks automatically.",
    "AI tools help users perform smart automated tasks.",
    "Cloud computing means storing and accessing data over the internet.",
    "Cybersecurity protects systems and data from digital attacks.",
    "Robotics combines machines with intelligent programming.",
    "An algorithm is a step-by-step method to solve problems.",
    "A database stores and organizes information digitally.",
    "Yes, AI can generate images using AI models.",
    "Yes, AI can help write and understand code."
]
print("FAQ Chatbot Started!")
print("Type 'bye' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break

    # Combine questions with user input
    all_questions = questions + [user_input]

    # Convert text into vectors
    vectorizer = CountVectorizer().fit_transform(all_questions)

    # Calculate similarity
    similarity = cosine_similarity(vectorizer[-1], vectorizer[:-1])

        # Find best match
    index = np.argmax(similarity)

    # Similarity score
    score = similarity[0][index]

    # Check confidence
    if score < 0.6:
        print("Bot: Sorry, I do not understand your question.")
    else:
        print("Bot:", answers[index])