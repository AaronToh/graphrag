"""
Example usage of the RAG evaluation module with OpenRouter.

Run this file to see an example of using OpenRouter to access various LLM models
for RAG evaluation.
"""

import pandas as pd
from haystack import Document
from eval import evaluate_rag_pipeline

# Sample test cases with realistic retrieved documents
test_cases = [
    (
        "What is machine learning?",
        [
            Document(
                content="Machine learning is a subset of artificial intelligence that enables systems to learn from data. "
                "Rather than being explicitly programmed to perform a task, machine learning algorithms build models "
                "based on sample data, known as training data, to make predictions or decisions without being "
                "explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, "
                "such as email filtering and computer vision, where it is difficult or unfeasible to develop "
                "conventional algorithms to perform the needed tasks.",
                meta={"source": "ml_intro.txt", "score": 0.95},
            ),
            Document(
                content="It uses algorithms to find patterns in data without explicit programming. "
                "The three main types of machine learning are supervised learning, unsupervised learning, "
                "and reinforcement learning. Supervised learning uses labeled data to train models, "
                "unsupervised learning finds hidden patterns in unlabeled data, and reinforcement learning "
                "learns through trial and error with rewards and penalties.",
                meta={"source": "ml_types.txt", "score": 0.88},
            ),
            Document(
                content="Machine learning has evolved significantly since the 1950s. Arthur Samuel coined the term "
                "in 1959, defining it as a field of study that gives computers the ability to learn without "
                "being explicitly programmed. Modern machine learning techniques include decision trees, "
                "random forests, support vector machines, and neural networks.",
                meta={"source": "ml_history.txt", "score": 0.82},
            ),
            Document(
                content="Common applications of machine learning include recommendation systems (like Netflix and Spotify), "
                "fraud detection in banking, medical diagnosis, autonomous vehicles, and natural language processing. "
                "These applications rely on the ability of ML algorithms to recognize patterns and make decisions "
                "based on data.",
                meta={"source": "ml_applications.txt", "score": 0.75},
            ),
        ],
        "Machine learning is a method where computers learn from data using algorithms.",
    ),
    (
        "What is deep learning?",
        [
            Document(
                content="Deep learning is a type of machine learning that uses neural networks with multiple layers. "
                "These deep neural networks, also called deep nets, are composed of an input layer, multiple hidden layers, "
                "and an output layer. Each layer contains nodes (or neurons) that process information and pass it to the next layer. "
                "The 'deep' in deep learning refers to the number of hidden layers in the network, which can range from "
                "a few to hundreds of layers.",
                meta={"source": "deep_learning_basics.txt", "score": 0.92},
            ),
            Document(
                content="These networks can learn hierarchical representations of data. "
                "Lower layers typically learn simple features (like edges in images), while deeper layers combine these "
                "to recognize more complex patterns (like faces or objects). This hierarchical feature learning is what "
                "makes deep learning particularly effective for tasks like image recognition, speech recognition, and "
                "natural language understanding.",
                meta={"source": "hierarchical_learning.txt", "score": 0.89},
            ),
            Document(
                content="Popular deep learning architectures include Convolutional Neural Networks (CNNs) for image processing, "
                "Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks for sequential data, "
                "and Transformers for natural language processing. CNNs excel at spatial pattern recognition, "
                "while RNNs are designed to handle sequential information with temporal dependencies.",
                meta={"source": "dl_architectures.txt", "score": 0.85},
            ),
            Document(
                content="Training deep learning models requires large amounts of data and computational power. "
                "GPUs (Graphics Processing Units) have become essential for deep learning because they can perform "
                "the massive parallel computations needed to train deep networks efficiently. Modern frameworks like "
                "TensorFlow, PyTorch, and Keras have made it easier to build and train deep learning models.",
                meta={"source": "dl_training.txt", "score": 0.78},
            ),
            Document(
                content="Deep learning has achieved remarkable success in recent years. It powers virtual assistants like "
                "Siri and Alexa, enables self-driving cars to recognize objects and navigate roads, and has achieved "
                "superhuman performance in games like Go and chess. In healthcare, deep learning models can detect "
                "diseases from medical images with accuracy comparable to expert physicians.",
                meta={"source": "dl_applications.txt", "score": 0.73},
            ),
        ],
        "Deep learning uses multi-layered neural networks to learn complex patterns.",
    ),
    (
        "How does natural language processing work?",
        [
            Document(
                content="Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on the "
                "interaction between computers and human language. NLP combines computational linguistics with machine learning "
                "and deep learning to enable computers to understand, interpret, and generate human language in a valuable way. "
                "The goal is to bridge the gap between human communication and computer understanding.",
                meta={"source": "nlp_overview.txt", "score": 0.94},
            ),
            Document(
                content="NLP tasks typically involve several stages: tokenization (breaking text into words or subwords), "
                "part-of-speech tagging (identifying nouns, verbs, etc.), named entity recognition (identifying people, places, organizations), "
                "parsing (analyzing grammatical structure), and semantic analysis (understanding meaning). Modern NLP systems "
                "often use deep learning models, particularly transformers like BERT and GPT, which can capture context "
                "and nuances in language.",
                meta={"source": "nlp_pipeline.txt", "score": 0.91},
            ),
            Document(
                content="Common NLP applications include machine translation (like Google Translate), sentiment analysis "
                "(determining if text is positive or negative), chatbots and virtual assistants, text summarization, "
                "and question answering systems. These applications rely on understanding both the syntax (structure) "
                "and semantics (meaning) of language.",
                meta={"source": "nlp_apps.txt", "score": 0.84},
            ),
            Document(
                content="Word embeddings are a key technique in NLP that represent words as dense vectors in a high-dimensional space. "
                "Similar words have similar vector representations, allowing models to understand semantic relationships. "
                "Techniques like Word2Vec, GloVe, and contextual embeddings from transformers have revolutionized how "
                "machines process language.",
                meta={"source": "word_embeddings.txt", "score": 0.79},
            ),
        ],
        "Natural language processing uses computational techniques to enable computers to understand and process human language.",
    ),
]

# Optional ground truth for additional metrics
ground_truth_answers = [
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "Deep learning is a type of machine learning using neural networks with multiple layers.",
    "Natural language processing uses computational techniques to enable computers to understand and process human language.",
]

# Ground truth documents for Document MRR evaluation
# These represent the ideal/relevant documents that should be retrieved for each question
ground_truth_documents = [
    # Ground truth for "What is machine learning?"
    [
        Document(
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data. "
            "Rather than being explicitly programmed to perform a task, machine learning algorithms build models "
            "based on sample data, known as training data, to make predictions or decisions without being "
            "explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, "
            "such as email filtering and computer vision, where it is difficult or unfeasible to develop "
            "conventional algorithms to perform the needed tasks.",
            meta={"source": "ml_intro.txt"},
        ),
        Document(
            content="It uses algorithms to find patterns in data without explicit programming. "
            "The three main types of machine learning are supervised learning, unsupervised learning, "
            "and reinforcement learning. Supervised learning uses labeled data to train models, "
            "unsupervised learning finds hidden patterns in unlabeled data, and reinforcement learning "
            "learns through trial and error with rewards and penalties.",
            meta={"source": "ml_types.txt"},
        ),
    ],
    # Ground truth for "What is deep learning?"
    [
        Document(
            content="Deep learning is a type of machine learning that uses neural networks with multiple layers. "
            "These deep neural networks, also called deep nets, are composed of an input layer, multiple hidden layers, "
            "and an output layer. Each layer contains nodes (or neurons) that process information and pass it to the next layer. "
            "The 'deep' in deep learning refers to the number of hidden layers in the network, which can range from "
            "a few to hundreds of layers.",
            meta={"source": "deep_learning_basics.txt"},
        ),
        Document(
            content="These networks can learn hierarchical representations of data. "
            "Lower layers typically learn simple features (like edges in images), while deeper layers combine these "
            "to recognize more complex patterns (like faces or objects). This hierarchical feature learning is what "
            "makes deep learning particularly effective for tasks like image recognition, speech recognition, and "
            "natural language understanding.",
            meta={"source": "hierarchical_learning.txt"},
        ),
    ],
    # Ground truth for "How does natural language processing work?"
    [
        Document(
            content="Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on the "
            "interaction between computers and human language. NLP combines computational linguistics with machine learning "
            "and deep learning to enable computers to understand, interpret, and generate human language in a valuable way. "
            "The goal is to bridge the gap between human communication and computer understanding.",
            meta={"source": "nlp_overview.txt"},
        ),
        Document(
            content="NLP tasks typically involve several stages: tokenization (breaking text into words or subwords), "
            "part-of-speech tagging (identifying nouns, verbs, etc.), named entity recognition (identifying people, places, organizations), "
            "parsing (analyzing grammatical structure), and semantic analysis (understanding meaning). Modern NLP systems "
            "often use deep learning models, particularly transformers like BERT and GPT, which can capture context "
            "and nuances in language.",
            meta={"source": "nlp_pipeline.txt"},
        ),
    ],
]


def main():
    """Example: Using OpenRouter to evaluate RAG pipeline"""
    print("\n" + "=" * 60)
    print("RAG Evaluation with OpenRouter")
    print("=" * 60)
    print("\nNote: Set OPENROUTER_API_KEY environment variable")

    results = evaluate_rag_pipeline(
        test_cases=test_cases,
        ground_truth_answers=ground_truth_answers,
        ground_truth_documents=ground_truth_documents,
        faithfulness_llm_provider="openrouter",
        faithfulness_llm_model="openai/gpt-4o-mini-2024-07-18",
    )

    print("\n" + "=" * 60)
    print("Aggregated Metrics")
    print("=" * 60)

    # Pretty print aggregated metrics
    agg_report = results["aggregated_report"]
    if (
        isinstance(agg_report, dict)
        and "metrics" in agg_report
        and "score" in agg_report
    ):
        print(f"\n{'Metric':<25} {'Score':>10}")
        print("-" * 37)
        for metric, score in zip(agg_report["metrics"], agg_report["score"]):
            print(f"{metric:<25} {score:>10.4f}")
    else:
        print(agg_report)

    print("\n" + "=" * 60)
    print("Detailed Results")
    print("=" * 60)

    # Pretty print detailed results with pandas
    print(results["detailed_results"])

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
