from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric, FaithfulnessMetric

# Setting environment variables from .env
load_dotenv()
OUT_FILE = "./output/deepeval_metrics.csv"

# This model judges the RAGs responses
JUDGE = GPTModel(
    model="gpt-5-chat-latest", 
    temperature=0,       
)


def test_recall_precision_relevancy_at_k():
    # In the real CI/CD pipeline, these data are retrieved from the vector store.
    # Here, this is a fixed set, so that the reproducibility could also be tested as a POC,
    # and the related metrics could easily be checked.
    retrieved_docs = [
            "Opel is a car.",
            "The price of Opel Astra depends on the regional markets.",
            "The Opel Astra price is about 9M HUF in the 2024 price listings.",
            "Its luggage rack is 300 liters",
            "The Gellert Hill is in Budapest.",
        ]

    sample_queries = [
        "How much does an Opel Astra cost?",
        "How big is the luggage rack of an Opel Astra?"
    ]

    expected_responses = [
        "The Opel Astra price is about 9M HUF in the 2024 price listings.",
        "The luggage rack of an Opel Astra amounts 300 liters."
    ]

    actual_responses = [
        "The Opel Astra costs around 9M HUF according to 2024 listings.",
        "The luggage rack is about 300 liters."
    ]

    rows = []
    for query, reference, answer in zip(sample_queries,expected_responses, actual_responses):

        recall_metric = ContextualRecallMetric(model=JUDGE)
        precision_metric = ContextualPrecisionMetric(model=JUDGE)
        relevancy_metric = ContextualRelevancyMetric(model=JUDGE)
        faithfulness_metric = FaithfulnessMetric(model=JUDGE)

        # Build the test case ---
        tc = LLMTestCase(
            input=query,
            retrieval_context=retrieved_docs,
            expected_output=reference,
            actual_output=answer
        )

        # Metrics
        # Contextual Recall  = (number of ground-truth claims supported by retrieved contexts)/ (total number of ground-truth claims)
        #        It shows to which extent the ground truth claims were supported by the retrieved context

        # Contextual Precision (ranking based) = average( number of relevant chunks up to rank k / k )
        #        It shows how clean the retrieval was with regard to the ranking of the retrieved chunks 
        #        If there is 1 relevant chunk of 4 but the relevant chunk is at the first position, then precision = 1
        #        If there is 1 relevant chunk of 4 but the relevant chunk is at the third position, then precision = 1/3

        # Contextual Relevancy = (relevant document chunks with regard to the query)/(all the retrieved chunks, i.e. K)
        #        It shows the semantic relevancy of the retrieved document chunks with regard to the query, i.e. how much percent of the retrieved chunks are relevant for the query.

        # Faithfulness = (number of supported claims in the actual answer with regard to the retrieved document chunks) / (total number of claims in the actual answer) 
        #       It shows hallucination or unsupported claims with regard to the retrieved data.
        #       If the actual answer has 3 claims but only 2 of them can be supported by the retrieved document chunks, then faithfulness = 2/3

        # FactualCorrectness: No factual correctness metric is available in DeepEval but a custom metric can be set up. (Different compared to the RAGAS metric!)

        # Measure and report ---
        K = len(retrieved_docs)
        recall_metric.measure(tc)
        precision_metric.measure(tc)
        relevancy_metric.measure(tc)
        faithfulness_metric.measure(tc)

        print("=== Retrieval quality ===")
        print(f"Judge model:  {JUDGE.model_name}")
        print(f"K:            {K}")

        print(f"Contextual Recall@{K}:    {recall_metric.score:.3f}")
        reason = getattr(recall_metric, "reason", None)
        print(f"    Reason: {reason}")

        print(f"Contextual Precision@{K}: {precision_metric.score:.3f}")
        reason = getattr(precision_metric, "reason", None)
        print(f"    Reason: {reason}")

        print(f"Contextual Relevancy@{K}: {relevancy_metric.score:.3f}")
        reason = getattr(relevancy_metric, "reason", None)
        print(f"    Reason: {reason}")
        
        print(f"Faithfulness: {faithfulness_metric.score:.3f}")
        reason = getattr(faithfulness_metric, "reason", None)
        print(f"    Reason: {reason}")
        print("\n")
    
        # assertions
        # CI/CD quality gate for a possible CI/CD integration
        # assert recall_metric.score >= 0.7, "Contextual Recall below threshold"
        # assert precision_metric.score >= 0.5, "Contextual Precision below threshold"
        # assert relevancy_metric.score >= 0.5, "Contextual Relevancy below threshold"
        # assert faithfulness_metric.score >= 0.9, "Faithfulness below threshold"

        # collecting for pandas DataFrame
        rows.append({
            "query": query,
            "expected_answer": reference,
            "actual_answer": answer,
            "contextual_recall": recall_metric.score,
            "contextual_precision": precision_metric.score,
            "contextual_relevancy": relevancy_metric.score,
            "faithfulness": faithfulness_metric.score,
        })


    # building and saving the table
    df = pd.DataFrame(rows)

    # sorting by faithfulness (lowest first)
    df_sorted = df.sort_values(by="faithfulness", ascending=True)

    # ensure parent dirs exist and save
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(OUT_FILE, index=True)

    print("=== DeepEval metrics (sorted by faithfulness asc) ===")
    print(df_sorted)
    print(f"\nSaved CSV to: {OUT_FILE}")
