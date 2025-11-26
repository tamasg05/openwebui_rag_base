from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ContextPrecision
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper 
from langchain_openai import ChatOpenAI

from dataclasses import dataclass, field
from typing import Dict, Set
import json

from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType

OUT_FILE = "./output/ragas_per_row_metrics.json"
INPUT_FILE = "./input/demo_data.json"

MODEL = "gpt-5-chat-latest"

# Setting environment variables from .env
load_dotenv()

# Loading evaluation data from JSON file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    demo_data = json.load(f)

records = demo_data.get("records", [])

# OPENAI_API_KEY api key needs to be set as an environment variable, see .env file
llm = ChatOpenAI(model=MODEL, 
                 temperature=0,
                 request_timeout=120, # timeout in sec
                 max_retries=2,
                ) 
judge = LangchainLLMWrapper(llm)


# Building the dataset from JSON records
dataset = []

for rec in records:
    query = rec["query"]
    chunks = rec["chunks"]
    reference = rec["reference_response"]
    answer = rec["actual_response"]

    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": chunks,
            "response": answer,
            "reference": reference,
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

# ContextualRelevancy: Not available in RAGAS but a custom evaluator can be set up. (Different compared to DeepEval, and Promptfoo.)
#       See the below custom metric definition. 
#       Contextual relevancy = (number of relevant chunks with regard to the query) / (total number of retrieved chunks, i.e. K)

# Faithfulness = (number of supported claims in the actual answer with regard to the retrieved document chunks) / (total number of claims in the actual answer) 
#       It shows hallucination or unsupported claims with regard to the retrieved data.
#       If the actual answer has 3 claims but only 2 of them can be supported by the retrieved document chunks, then faithfulness = 2/3

# FactualCorrectness = F1 score = 2 * Precision * Recall / (Precision + Recall) with regard to the expected response as reference; harmonic mean penalizes models that do well on the one metric but poorly on the other.
#       The metric is also added for the factual precision and factual recall, see the code below.
#
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
#       

# If the dataset contains several queries, with the context chunks retrieved, the actual and the expected answers, then the evaluation is done individually, and finally the arithmetic mean is computed, if the result is printed as print(result).
# If the computation of one metric fails, an NaN value will appear (raise_exceptions=False) and the computation goes on.



# Custom metric definition:
# Pydantic models for the LLM prompt (for data types and validation)
# It is necessary for the custom metric definition: contextual relevancy

class AllChunksRelevancyInput(BaseModel):
    user_input: str = Field(description="The user query")
    chunks: list[str] = Field(
        description=(
            "List of retrieved document chunks. Each element is one chunk of text "
            "retrieved from the vector store."
        )
    )


class AllChunksRelevancyOutput(BaseModel):
    relevant_count: int = Field(
        description=(
            "The number of chunks that are clearly relevant and useful for answering "
            "the query. Must be an integer between 0 and len(chunks)."
        )
    )


class AllChunksRelevancyPrompt(PydanticPrompt[AllChunksRelevancyInput, AllChunksRelevancyOutput]):
    instruction = (
        "You are evaluating the retrieval step of a RAG system.\n"
        "Given a user query and a list of retrieved document chunks, determine how many "
        "of these chunks are relevant for answering the query.\n\n"
        "Guidelines:\n"
        "- A chunk is relevant if it directly helps to answer the query, even partially.\n"
        "- If a chunk is off-topic, only tangentially related, or does not help answer "
        "  the query, consider it not relevant.\n"
        "- Count how many chunks are relevant and return that number as 'relevant_count'.\n"
        "- 'relevant_count' must be an integer between 0 and len(chunks).\n"
    )
    input_model = AllChunksRelevancyInput
    output_model = AllChunksRelevancyOutput
    examples = [
        (
            AllChunksRelevancyInput(
                user_input="How much does an Opel Astra cost?",
                chunks=[
                    "Opel is a car.",
                    "The price of Opel Astra depends on the regional markets.",
                    "The Opel Astra price is about 9M HUF in the 2024 price listings.",
                    "Its luggage rack is 300 liters.",
                    "The Gellert Hill is in Budapest.",
                ],
            ),
            AllChunksRelevancyOutput(
                # Example judgment: the Opel Astra price-related chunks are relevant,
                # the luggage rack and Gellert Hill are not.
                relevant_count=2,
            ),
        ),
        (
            AllChunksRelevancyInput(
                user_input="Where is the Gellert Hill located?",
                chunks=[
                    "Opel is a car.",
                    "The Gellert Hill is in Budapest.",
                ],
            ),
            AllChunksRelevancyOutput(
                relevant_count=1,
            ),
        ),
    ]


# The custom metric definition for Contextual Relevancy

@dataclass
class ContextualRelevancy(MetricWithLLM, SingleTurnMetric):
    """
    Contextual relevancy = (# relevant retrieved chunks) / (total retrieved chunks: K)

    - Uses only the query and the retrieved_contexts (retriever quality).
    - Single LLM call per sample: the model sees all chunks at once and returns
      'relevant_count'.
    """
    name: str = "contextual_relevancy"
    _required_columns: Dict[MetricType, Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "retrieved_contexts"}
        }
    )
    relevancy_prompt: PydanticPrompt = AllChunksRelevancyPrompt()

    async def _single_turn_ascore(self, sample, callbacks):
        contexts = sample.retrieved_contexts or []
        total = len(contexts)
        if total == 0:
            # no chunks retrieved -> relevancy is defined as 0
            return 0.0

        prompt_input = AllChunksRelevancyInput(
            user_input=sample.user_input,
            chunks=contexts,
        )

        result = await self.relevancy_prompt.generate(
            data=prompt_input,
            llm=self.llm,
        )

        # Safety: clamp to [0, total]
        relevant_count = max(0, min(result.relevant_count, total))

        return relevant_count / total



result = evaluate(dataset=evaluation_dataset,
                  metrics=[LLMContextRecall(), 
                           ContextualRelevancy(),
                           Faithfulness(), 
                           FactualCorrectness(name="factual_precision", mode="precision"),
                           FactualCorrectness(name="factual_recall",mode="recall"),
                           FactualCorrectness()],
                  llm=judge,
                  raise_exceptions=False)

# Converting the individual lines to a table
df = result.to_pandas()

# Dropping the retrieved context data for better legibility
df = df.drop(columns=["retrieved_contexts"])

# sorting by factual correctness, lowest first
df_sorted = df.sort_values(by="factual_correctness(mode=f1)", ascending=True)

# Saving the table as JSON
# Ensuring parent directories exist
Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)

# Exporting as a list of records: [{col: value, ...}, ...]
df_sorted.to_json(
    OUT_FILE,
    orient="records",
    force_ascii=False,
    indent=2
)

print(df_sorted)
print(result)