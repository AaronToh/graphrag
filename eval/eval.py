"""
RAG Pipeline Evaluation Module

Evaluate RAG pipelines using Document MRR, Faithfulness, and Semantic Answer Similarity.
For detailed documentation, see evaluation.md
"""

import os
from typing import List, Tuple, Dict, Any, Optional, Union
from haystack import Pipeline, Document
from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
from haystack.components.evaluators.sas_evaluator import SASEvaluator
from haystack.components.evaluators import AnswerExactMatchEvaluator
from haystack.evaluation.eval_run_result import EvaluationRunResult
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret


def create_faithfulness_evaluator(
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_base_url: Optional[str] = None,
    api_key: Optional[Union[Secret, str]] = None,
) -> FaithfulnessEvaluator:
    """Create a FaithfulnessEvaluator with custom LLM configuration.

    Args:
        llm_provider: The LLM provider to use. Options: "openai", "ollama", "openrouter"
        model: The model name to use.
            For OpenAI: "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", etc.
            For Ollama: "llama3.2", "mistral", "qwen2.5", etc.
            For OpenRouter: "meta-llama/llama-3.2-3b-instruct", "anthropic/claude-3.5-sonnet", etc.
        api_base_url: Base URL for the API.
            For Ollama: typically "http://localhost:11434/v1"
            For OpenRouter: "https://openrouter.ai/api/v1"
            For OpenAI: uses default unless specified
        api_key: API key for authentication.
            For OpenAI: Required (or set OPENAI_API_KEY env var)
            For Ollama: Use "ollama" or any non-empty string
            For OpenRouter: Required (or set OPENROUTER_API_KEY env var)

    Returns:
        Configured FaithfulnessEvaluator instance
    """
    if llm_provider.lower() == "ollama":
        # Ollama configuration
        if api_base_url is None:
            api_base_url = "http://localhost:11434/v1"
        if api_key is None:
            api_key = Secret.from_token("ollama")

        chat_generator = OpenAIChatGenerator(
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            generation_kwargs={"response_format": {"type": "json_object"}},
        )
    elif llm_provider.lower() == "openrouter":
        # OpenRouter configuration
        if api_base_url is None:
            api_base_url = "https://openrouter.ai/api/v1"
        if api_key is None:
            # Try to get from environment variable
            api_key_str = os.getenv("OPENROUTER_API_KEY")
            if api_key_str:
                api_key = Secret.from_env_var("OPENROUTER_API_KEY")
            else:
                raise ValueError(
                    "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                    "or pass faithfulness_api_key parameter"
                )

        chat_generator = OpenAIChatGenerator(
            api_key=(
                api_key if isinstance(api_key, Secret) else Secret.from_token(api_key)
            ),
            api_base_url=api_base_url,
            model=model,
            generation_kwargs={"response_format": {"type": "json_object"}},
        )
    elif llm_provider.lower() == "openai":
        # OpenAI configuration
        kwargs = {
            "model": model,
            "generation_kwargs": {"response_format": {"type": "json_object"}},
        }
        if api_key is not None:
            kwargs["api_key"] = (
                api_key if isinstance(api_key, Secret) else Secret.from_token(api_key)
            )
        if api_base_url is not None:
            kwargs["api_base_url"] = api_base_url

        chat_generator = OpenAIChatGenerator(**kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {llm_provider}. Use 'openai', 'ollama', or 'openrouter'"
        )

    return FaithfulnessEvaluator(chat_generator=chat_generator)


def evaluate_rag_pipeline(
    test_cases: List[Tuple[str, List[Document], str]],
    ground_truth_answers: Optional[List[str]] = None,
    ground_truth_documents: Optional[List[List[Document]]] = None,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    run_name: str = "rag_evaluation",
    faithfulness_llm_provider: str = "openai",
    faithfulness_llm_model: str = "gpt-4o-mini",
    faithfulness_api_base_url: Optional[str] = None,
    faithfulness_api_key: Optional[Union[Secret, str]] = None,
) -> Dict[str, Any]:
    """Evaluate RAG pipeline results using multiple metrics.

    See evaluation.md for detailed documentation and examples.
    """

    # Extract components from test cases
    questions = []
    retrieved_docs = []
    predicted_answers = []
    contexts = []

    for question, docs, answer in test_cases:
        questions.append(question)
        retrieved_docs.append(docs)
        predicted_answers.append(answer)
        # Extract content from documents for faithfulness evaluation
        contexts.append([doc.content for doc in docs])

    # Build evaluation pipeline
    eval_pipeline = Pipeline()

    # Add evaluators based on available data
    evaluators_to_run = {}

    # # Always add faithfulness evaluator (doesn't require ground truth)
    # faithfulness_evaluator = create_faithfulness_evaluator(
    #     llm_provider=faithfulness_llm_provider,
    #     model=faithfulness_llm_model,
    #     api_base_url=faithfulness_api_base_url,
    #     api_key=faithfulness_api_key,
    # )
    # eval_pipeline.add_component("faithfulness", faithfulness_evaluator)
    # evaluators_to_run["faithfulness"] = {
    #     "questions": questions,
    #     "contexts": contexts,
    #     "predicted_answers": predicted_answers,
    # }

    # Add MRR evaluator if ground truth documents are provided
    if ground_truth_documents is not None:
        eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
        evaluators_to_run["doc_mrr_evaluator"] = {
            "ground_truth_documents": ground_truth_documents,
            "retrieved_documents": retrieved_docs,
        }

    # Add SAS evaluator if ground truth answers are provided
    if ground_truth_answers is not None:
        eval_pipeline.add_component("sas_evaluator", SASEvaluator(model=model))
        evaluators_to_run["sas_evaluator"] = {
            "predicted_answers": predicted_answers,
            "ground_truth_answers": ground_truth_answers,
        }

    # Add AnswerExactMatch evaluator if ground truth answers are provided
    if ground_truth_answers is not None:
        eval_pipeline.add_component("answer_exact_match", AnswerExactMatchEvaluator())
        evaluators_to_run["answer_exact_match"] = {
            "predicted_answers": predicted_answers,
            "ground_truth_answers": ground_truth_answers,
        }

    # Run evaluation pipeline
    results = eval_pipeline.run(evaluators_to_run)

    # Prepare inputs for EvaluationRunResult
    inputs = {
        "question": questions,
        "predicted_answer": predicted_answers,
    }

    # Add optional inputs if available
    if ground_truth_answers is not None:
        inputs["answer"] = ground_truth_answers

    if ground_truth_documents is not None:
        # Flatten ground truth documents for display
        inputs["ground_truth_docs"] = [
            [doc.content for doc in docs] for docs in ground_truth_documents
        ]

    # Create evaluation result object
    evaluation_result = EvaluationRunResult(
        run_name=run_name, inputs=inputs, results=results
    )

    # Generate reports
    aggregated_report = evaluation_result.aggregated_report()
    detailed_results = evaluation_result.detailed_report(output_format="df")

    return {
        "aggregated_report": aggregated_report,
        "detailed_results": detailed_results,
        "eval_pipeline_results": results,
        "evaluation_result": evaluation_result,
    }


def evaluate_with_defaults(
    questions: List[str],
    retrieved_documents: List[List[Document]],
    predicted_answers: List[str],
    ground_truth_answers: Optional[List[str]] = None,
    ground_truth_documents: Optional[List[List[Document]]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience wrapper accepting separate lists instead of tuples.

    See evaluation.md for detailed documentation and examples.
    """
    # Validate input lengths
    if not (len(questions) == len(retrieved_documents) == len(predicted_answers)):
        raise ValueError(
            "questions, retrieved_documents, and predicted_answers must have the same length"
        )

    if ground_truth_answers is not None and len(ground_truth_answers) != len(questions):
        raise ValueError("ground_truth_answers must have the same length as questions")

    if ground_truth_documents is not None and len(ground_truth_documents) != len(
        questions
    ):
        raise ValueError(
            "ground_truth_documents must have the same length as questions"
        )

    # Combine into test cases
    test_cases = list(zip(questions, retrieved_documents, predicted_answers))

    return evaluate_rag_pipeline(
        test_cases=test_cases,
        ground_truth_answers=ground_truth_answers,
        ground_truth_documents=ground_truth_documents,
        **kwargs,
    )


if __name__ == "__main__":
    print("RAG Evaluation Module")
    print("=" * 50)
    print("\nFor detailed documentation, see evaluation.md")
    print("\nQuick example:")
    print(
        """
from haystack import Document
from eval import evaluate_rag_pipeline

test_cases = [
    ("What is ML?", [Document(content="ML is...")], "Machine learning is...")
]
results = evaluate_rag_pipeline(test_cases=test_cases)
print(results["aggregated_report"])
    """
    )
