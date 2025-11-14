from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ContextPrecision
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper 
from langchain_openai import ChatOpenAI

OUT_FILE = "./output/ragas_per_row_metrics.csv"
MODEL = "gpt-5-chat-latest"

# Setting environment variables from .env
load_dotenv()

# OPENAI_API_KEY api key needs to be set as an environment variable, see .env file
llm = ChatOpenAI(model=MODEL, temperature=0) 
judge = LangchainLLMWrapper(llm)

# In the real CI/CD pipeline, these data are retrieved from the vector store.
# Here, this is a fixed set, so that the reproducibility could also be tested as a POC,
# and the related metrics could easily be checked.
retrieved_docs = [
        "Opel is a car.",
        "The price of Opel Astra depends on the regional markets.",
        "The Opel Astra price is about 9M HUF in the 2024 price listings.",
        "Its luggage rack is 300 liters.",
        "The Gellert Hill is in Budapest."
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

dataset = []

for query, reference, answer in zip(sample_queries,expected_responses, actual_responses):

    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":retrieved_docs,
            "response":answer,
            "reference":reference
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)

# Metrics
# ContextRecall  = (number of claims supported by the retrieved document chunks in the expected response)/ (total number of claims in the expected response)
#        It shows to which extent the ground truth claims were supported by the retrieved document chunks

# ContextPrecision (ranking based) = average( number of relevant chunks up to rank k / k )
#        It shows how clean the retrieval was with regard to the ranking of the retrieved chunks 
#        If there is 1 relevant chunk of 4 but the relevant chunk is at the first position, then precision = 1
#        If there is 1 relevant chunk of 4 but the relevant chunk is at the third position, then precision = 1/3

# ContextRelevancy: Not available in RAGAS but a custom evaluator can be set up. (Different compared to DeepEval, and Promptfoo.)

# Faithfulness = (number of supported claims in the actual answer with regard to the retrieved document chunks) / (total number of claims in the actual answer) 
#       It shows hallucination or unsupported claims with regard to the retrieved data.
#       If the actual answer has 3 claims but only 2 of them can be supported by the retrieved document chunks, then faithfulness = 2/3

# FactualCorrectness = F1 score = 2 * Precision * Recall / (Precision + Recall) with regard to the expected response as reference; harmonic mean penalizes models that do well on the one metric but poorly on the other.
#       It shows the correctness of information in the actual answer.
#
#       For each claim, an LLM compares them and counts:
#       TP (true positive): claims in both the model’s answer and the reference.
#       FP (false positive): claims in the model’s answer but not in the reference.
#       FN (false negative): claims in the reference but missing from the model’s answer.
#
#       Then ragas calculates one or more of:
#       Precision = TP / (TP + FP)
#       Recall    = TP / (TP + FN)

# If the dataset contains several queries, with the context chunks retrieved, the actual and the expected answers, then the evaluation is done individually, and finally the arithmetic mean is computed, if the result is printed as print(result).

# If the computation of one metric fails, an NaN value will appear (raise_exceptions=False) and the computation goes on.
result = evaluate(dataset=evaluation_dataset,
                  metrics=[LLMContextRecall(), 
                           ContextPrecision(), 
                           Faithfulness(), 
                           FactualCorrectness()],
                  llm=judge,
                  raise_exceptions=False)

# Converting the individual lines to a table
df = result.to_pandas()

# Dropping the retrieved context data for better legibility
df = df.drop(columns=["retrieved_contexts"])

# sorting by factual correctness, lowest first
df_sorted = df.sort_values(by="factual_correctness(mode=f1)", ascending=True)

# Saving the table
# Ensuring parent directories exist
Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
df_sorted.to_csv(OUT_FILE, index=True)

print(df_sorted)  
print(result)
