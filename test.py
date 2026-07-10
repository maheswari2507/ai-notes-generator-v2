from utils.transformer_summarizer import generate_summary

text = """
Artificial Intelligence (AI) is a branch of computer science that focuses on building intelligent machines.
Machine Learning is a subset of AI that enables computers to learn from data.
Deep Learning uses neural networks with multiple layers.
Natural Language Processing helps computers understand human language.
AI is used in healthcare, finance, education, and transportation.
AI can automate repetitive tasks and improve decision-making.
""" * 3

result = generate_summary(text)

print("\nSUMMARY:")
print(result["summary"])

print("\nKEY POINTS:")

for point in result["key_points"]:
    print("-", point)