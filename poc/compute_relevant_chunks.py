"""
Compute relevant chunks and LLM-based evaluation metrics for each query
using a single LLM call per query.

Expected input JSON structure per file:

Case A: top-level object with the following mandatory records:
{
  "records": [
    {
      "query": "...",
      "chunks": [ "...", ... ],
      "chunkIds": [ "...", ... ],
      "chunkScores": [ 0.76, 0.73, ... ],
      "reference_response": "...",
      "actual_response": "..."
    },
    ...
  ],
  "topK": ...,
  "similarityThreshold": ...,
  "temperature": ...,
  "model": "..."
}

Case B: a single record object:
{
  "query": "...",
  "chunks": [...],
  "chunkIds": [...],
  "chunkScores": [...],
  "reference_response": "...",
  "actual_response": "..."
}

Output (for each record):

{
  "query": "This is a query example.",
  "sum_of_relevance_scores": 4.35
  "relevance_scores": [score1, score2, score3, ...],
  "number_of_relevant_chunks": 3,
  "chunk_ids": ["id2", "id4", "id9"],
  "chunk_scores": [0.23, 0.7, 0.1],
  "chunks": ["chunk text 1", "chunk text 2", "chunk text 3", ...],
  "faithfulness": 0.87,
  "factual_recall": 0.92,
  "factual_precision": 0.75,
  "context_recall": 0.80
}
Requires: OPENAI_API_KEY in the environment.
"""

import argparse
import json
import time
import os
from typing import Any, Dict, List, Tuple
import csv
import statistics

from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
LLM_MODEL = "gpt-5.1-2025-11-13"  # or "gpt-4.1-mini"
LLM_TEMPERATURE = 0.0
SLEEP_SECONDS = 60  # sleep between input files to avoid rate limiting
RELEVANCE_THRESHOLD = 0.50  # a chunk is counted as relevant if the relevance score exceeds this limit

load_dotenv()


# ---------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------
def get_client() -> OpenAI:
    # OPENAI_API_KEY is read from env by default
    return OpenAI()


# ---------------------------------------------------------------------
# Computing quality metrics for RAG by the LLM
# Using relevance scores instead of binary relevant or not decisions
# ---------------------------------------------------------------------
def score_relevance_and_metrics(
    client: OpenAI,
    query: str,
    chunks: List[str],
    expected_answer: str,
    actual_answer: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> Tuple[List[float], float, float, float, float]:
    """
    Ask the LLM to:
      - assign a relevance score in [0,1] for EACH chunk,
      - compute faithfulness, factual_recall, factual_precision, context_recall.

    Returns:
        (relevance_scores, faithfulness, factual_recall, factual_precision, context_recall)
    """

    chunk_list_str_parts = []
    for idx, text in enumerate(chunks):
        chunk_list_str_parts.append(f"Chunk {idx}:\n{text}\n")
    chunk_list_str = "\n---\n".join(chunk_list_str_parts)

    system_msg = (
        "You are a strict relevance and factuality judge for a RAG system.\n"
        "You receive:\n"
        "- a user query,\n"
        "- a numbered list of retrieved document chunks,\n"
        "- an expected (reference) answer,\n"
        "- and an actual answer produced by another model.\n\n"
        "Your tasks:\n"
        "1) For EACH chunk (document fragment), assign a relevance score in [0,1] indicating how useful the chunk is\n"
        "   for answering the query.\n"
        "   - 1.00 = clearly contains direct information needed to answer the query.\n"
        "   - 0.00 = clearly irrelevant (cookie banners, navigation, boilerplate, unrelated topic).\n"
        "   - Values in between reflect uncertainty or partial usefulness.\n"
        "   - The closer the value is to 1.00, the more relevant the chunk is for answering the query.\n"
        "   - IMPORTANT: Return the chunk scores as a JSON object named \"relevance_by_index\".\n"
        "   - The keys must be 0-based chunk indices as STRINGS (for example: \"0\", \"1\", \"2\").\n"
        "   - The values must be numeric floats between 0.00 and 1.00.\n"
        "   - You MAY omit chunks whose score is exactly 0.00. Any omitted chunk will be treated as 0.00.\n"
        "   - Do NOT invent indices. Only use indices from 0 to N-1, where N is the number of chunks.\n"
        "   - Example: \"relevance_by_index\": {\"0\": 0.95, \"3\": 0.40, \"7\": 1.00}\n"
        "   - Be conservative: only use scores > 0.80 when the chunk is clearly helpful.\n\n"
        "2) Compute a FAITHFULNESS score (0 to 1) for the actual answer with respect to\n"
        "   the chunks: how grounded the answer is in those chunks.\n"
        "   - First, conceptually break the actual answer into atomic factual statements or\n"
        "     information units (short facts, claims, conditions, parameter values, etc.).\n"
        "   - For each statement, check whether it is supported by the content of the\n"
        "     chunks (it may be a paraphrase, but must not contradict the chunks).\n"
        "   - Let S = number of statements in the actual answer that ARE supported by\n"
        "     the chunks.\n"
        "   - Let T = total number of factual statements in the actual answer.\n"
        "   - Faithfulness is approximately S / T.\n"
        "   - If the answer is completely supported by the provided chunks, the score should\n"
        "     be 1.0.\n"
		"   - If T = 0 (no factual statements), return 0.00.\n"
        "   - If none of the statements are supported by the chunks (S = 0 and T > 0), return 0.00.\n\n"
        "3) Compute a FACTUAL_RECALL score (0 to 1) comparing the actual answer to the\n"
        "   expected (reference) answer.\n"
        "   - Break the EXPECTED answer into atomic information units: key factual\n"
        "     statements, conditions, parameters, definitions, or important details.\n"
        "   - Ignore purely stylistic elements, politeness, or minor wording differences.\n"
        "   - Let R = number of information units from the expected answer that are\n"
        "     correctly present (possibly paraphrased) in the actual answer.\n"
        "   - Let E = total number of important information units in the expected answer.\n"
        "   - Factual_recall is approximately R / E.\n"
        "   - 1.0 means all important information in the expected answer is present in\n"
        "     the actual answer.\n"
        "   - 0.0 means none of the important information from the expected answer is\n"
        "     present in the actual answer.\n"
		"   - If the expected (reference) answer is empty, missing, or only whitespace, return factual_recall = 0.00.\n\n"
        "4) Compute a FACTUAL_PRECISION score (0 to 1) comparing the actual answer to the\n"
        "   expected answer.\n"
        "   - Break the ACTUAL answer into atomic information units (as above).\n"
        "   - Let M = number of information units in the actual answer that ALSO appear\n"
        "     in the expected answer (possibly paraphrased, but factually equivalent).\n"
        "   - Let A = total number of factual information units in the actual answer.\n"
        "   - Factual_precision is approximately M / A.\n"
        "   - 1.0 means the actual answer does not add extra facts beyond\n"
        "     what is in the expected answer.\n"
        "   - 0.0 means the actual answer only contains additional/different information\n"
        "     and does not include the factual content of the expected answer.\n"
		"   - If the expected (reference) answer is empty, missing, or only whitespace, return factual_precision = 0.00.\n\n"
        "5) Compute a CONTEXT_RECALL score (0 to 1) for the chunks with respect\n"
        "   to the user query: how sufficient the chunks are to answer the query.\n"
        "   - Conceptually break the QUERY into its information requirements:\n"
        "     sub-questions, conditions, parameters, or key aspects that must be\n"
        "     addressed to give a complete and correct answer.\n"
        "   - Let C_total = number of important information requirements in the query.\n"
        "   - Let C_covered = number of those requirements for which the chunks\n"
        "     contain enough information to answer that part correctly (even if the\n"
        "     actual answer did not fully use it).\n"
        "   - Context_recall is approximately C_covered / C_total.\n"
        "   - 1.0 means the chunks, taken together, contain enough information\n"
        "     to fully answer the query.\n"
        "   - 0.0 means the chunks do not provide useful information for any\n"
        "     important part of the query.\n\n"
        "General instructions for all metrics:\n"
        "- Focus on factual content and conditions; ignore purely stylistic differences,\n"
        "  politeness, formatting, or minor reordering.\n"
        "- If something is ambiguous, choose a reasonable value between 0 and 1 that reflects\n"
        "  partial coverage (for recall/context_recall) or partial alignment (for precision).\n"
        "- All scores MUST be numeric floats between 0.0 and 1.0 (not percentages).\n"
        "- ROUND ALL SCORES to TWO DECIMAL PLACES (e.g., 0.73, 0.00, 1.00).\n"
        "- When in doubt, be slightly conservative rather than overly generous.\n\n"
        "Return ONLY valid JSON in the following form:\n"
        "{\n"
        "  \"relevance_by_index\": {\"0\": 0.0, \"1\": 0.75, \"2\": 0.1},\n"
        "  \"faithfulness\": 0.00-1.00,\n"
        "  \"factual_recall\": 0.00-1.00,\n"
        "  \"factual_precision\": 0.00-1.00,\n"
        "  \"context_recall\": 0.00-1.00\n"
        "}\n"
        "Rules for relevance_by_index:\n"
        "- Keys are 0-based chunk indices as strings.\n"
        "- Values are floats in [0,1].\n"
        "- You MAY omit indices whose score is exactly 0.0.\n"
        "- If an index is omitted, it will be treated as 0.0.\n"
        "- Do not include any key outside 0..N-1.\n"    )
		

    user_msg = (
        f"Query:\n{query}\n\n"
        "Here is the numbered list of chunks:\n\n"
        f"{chunk_list_str}\n\n"
        "Expected (reference) answer:\n"
        f"{expected_answer}\n\n"
        "Actual answer:\n"
        f"{actual_answer}\n\n"
        "Score each chunk's relevance and compute the metrics.\n"
        "Respond ONLY with JSON in the specified format."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = response.choices[0].message.content.strip()

    # Defaults
    relevance_scores: List[float] = [0.0] * len(chunks)
    faithfulness = factual_recall = factual_precision = context_recall = 0.0

    try:
        data = json.loads(content)

        # relevance_scores
        # Default dense vector (all chunks irrelevant unless explicitly scored)
        relevance_scores = [0.0] * len(chunks)

        # Preferred format: sparse map { "0": 0.95, "3": 0.4, ... }
        raw_map = data.get("relevance_by_index", None)

        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                try:
                    idx = int(k)   # keys are strings in JSON
                    if 0 <= idx < len(chunks):
                        score = float(v)
                        score = max(0.0, min(1.0, score))
                        relevance_scores[idx] = round(score, 2)
                    else:
                        print(f"### Invalid index out of range in relevance_by_index: {k}")
                except Exception:
                    print(f"### Invalid relevance_by_index entry: {k}: {v}")
        else:
            print("### No valid relevance scores returned; using all zeros")

        faithfulness = get_score(data, "faithfulness")
        factual_recall = get_score(data, "factual_recall")
        factual_precision = get_score(data, "factual_precision")
        context_recall = get_score(data, "context_recall")

    except json.JSONDecodeError:
        pass

    return relevance_scores, faithfulness, factual_recall, factual_precision, context_recall


# ---------------------------------------------------------------------
# Process a single record
# ---------------------------------------------------------------------
def process_record(client: OpenAI, record: Dict[str, Any]) -> Dict[str, Any]:
    query = record["query"]
    chunks: List[str] = record["chunks"]
    chunk_ids: List[Any] = record["chunkIds"]
    chunk_scores: List[float] = record["chunkScores"]
    expected_answer: str = record.get("reference_response", "")
    actual_answer: str = record.get("actual_response", "")

    if not (len(chunks) == len(chunk_ids) == len(chunk_scores)):
        raise ValueError("Length mismatch between chunks, chunkIds and chunkScores")

    (
        relevance_scores,
        faithfulness,
        factual_recall,
        factual_precision,
        context_recall,
    ) = score_relevance_and_metrics(
        client, query, chunks, expected_answer, actual_answer
    )

    relevant_indices = [i for i, s in enumerate(relevance_scores) if s >= RELEVANCE_THRESHOLD]

    sum_of_relevance_scores = round(sum(relevance_scores), 2)

    result = {
        "query": query,
        "sum_of_relevance_scores": sum_of_relevance_scores,

        # one score per chunk, aligned with chunkIds/chunks
        "relevance_scores": relevance_scores,

        # keep old-style summary (now deterministic for a fixed threshold)
        "number_of_relevant_chunks": len(relevant_indices),
        "chunk_ids": [chunk_ids[i] for i in relevant_indices],
        "chunk_scores": [chunk_scores[i] for i in relevant_indices],
        "chunks": [chunks[i] for i in relevant_indices],

        "faithfulness": faithfulness,
        "factual_recall": factual_recall,
        "factual_precision": factual_precision,
        "context_recall": context_recall,
    }

    return result

# ---------------------------------------------------------------------
# helper methods
# ---------------------------------------------------------------------
def get_score(data: dict, key: str) -> float:
    try:
        val = float(data.get(key, 0.0))
        return round(max(0.0, min(1.0, val)), 2)
    except Exception:
        return 0.0

def load_input(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_output(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_run_to_csv(
    input_path: str,
    results: List[Dict[str, Any]],
    csv_path: Path,
    temperature: Any = None,
    topk: Any = None,
    similarity_threshold: Any = None,
    model: Any = None,
) -> None:
    """
    Append one row to a CSV file with aggregated stats.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    n = max(1, len(results))  # avoid division by zero, though len(results) >= 1 in practice

    # ------------------------------------------------------------------
    # Collect per-record values into lists
    # ------------------------------------------------------------------
    relevant_chunks_list = [r.get("number_of_relevant_chunks", 0) for r in results]
    sum_of_relevance_scores_list = [r.get("sum_of_relevance_scores", 0.0) for r in results]

    faithfulness_list = [r.get("faithfulness", 0.0) for r in results]
    factual_recall_list = [r.get("factual_recall", 0.0) for r in results]
    factual_precision_list = [r.get("factual_precision", 0.0) for r in results]
    context_recall_list = [r.get("context_recall", 0.0) for r in results]

    # ------------------------------------------------------------------
    # Sums
    # ------------------------------------------------------------------
    total_relevant_chunks = sum(relevant_chunks_list)
    sum_relevance_scores_total = sum(sum_of_relevance_scores_list)

    sum_faithfulness = sum(faithfulness_list)
    sum_factual_recall = sum(factual_recall_list)
    sum_factual_precision = sum(factual_precision_list)
    sum_context_recall = sum(context_recall_list)

    # ------------------------------------------------------------------
    # Averages
    # ------------------------------------------------------------------
    avg_relevant_chunks = total_relevant_chunks / n
    avg_sum_of_relevance_scores = sum_relevance_scores_total / n

    avg_faithfulness = sum_faithfulness / n
    avg_factual_recall = sum_factual_recall / n
    avg_factual_precision = sum_factual_precision / n
    avg_context_recall = sum_context_recall / n

    # ------------------------------------------------------------------
    # Medians
    # ------------------------------------------------------------------
    median_relevant_chunks = statistics.median(relevant_chunks_list)
    median_sum_of_relevance_scores = statistics.median(sum_of_relevance_scores_list)

    median_faithfulness = statistics.median(faithfulness_list)
    median_factual_recall = statistics.median(factual_recall_list)
    median_factual_precision = statistics.median(factual_precision_list)
    median_context_recall = statistics.median(context_recall_list)

    # ------------------------------------------------------------------
    # Standard deviations (population std, pstdev -> 0.0 if n == 1)
    # ------------------------------------------------------------------
    std_relevant_chunks = statistics.pstdev(relevant_chunks_list)
    std_sum_of_relevance_scores = statistics.pstdev(sum_of_relevance_scores_list)

    std_faithfulness = statistics.pstdev(faithfulness_list)
    std_factual_recall = statistics.pstdev(factual_recall_list)
    std_factual_precision = statistics.pstdev(factual_precision_list)
    std_context_recall = statistics.pstdev(context_recall_list)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "date",
                    "input_file",
                    "temperature",
                    "topK",
                    "similarityThreshold",
                    "model",
                    "total_relevant_chunks",
                    "avg_relevant_chunks",
                    "median_relevant_chunks",
                    "std_relevant_chunks",
                    "sum_sum_of_relevance_scores",
                    "avg_sum_of_relevance_scores",
                    "median_sum_of_relevance_scores",
                    "std_sum_of_relevance_scores",
                    "sum_faithfulness",
                    "avg_faithfulness",
                    "median_faithfulness",
                    "std_faithfulness",
                    "sum_factual_recall",
                    "avg_factual_recall",
                    "median_factual_recall",
                    "std_factual_recall",
                    "sum_factual_precision",
                    "avg_factual_precision",
                    "median_factual_precision",
                    "std_factual_precision",
                    "sum_context_recall",
                    "avg_context_recall",
                    "median_context_recall",
                    "std_context_recall",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                input_path,
                temperature,
                topk,
                similarity_threshold,
                model,
                total_relevant_chunks,
                round(avg_relevant_chunks, 2),
                round(median_relevant_chunks, 2),
                round(std_relevant_chunks, 2),
                round(sum_relevance_scores_total, 2),
                round(avg_sum_of_relevance_scores, 2),
                round(median_sum_of_relevance_scores, 2),
                round(std_sum_of_relevance_scores, 2),
                round(sum_faithfulness, 2),
                round(avg_faithfulness, 2),
                round(median_faithfulness, 2),
                round(std_faithfulness, 2),
                round(sum_factual_recall, 2),
                round(avg_factual_recall, 2),
                round(median_factual_recall, 2),
                round(std_factual_recall, 2),
                round(sum_factual_precision, 2),
                round(avg_factual_precision, 2),
                round(median_factual_precision, 2),
                round(std_factual_precision, 2),
                round(sum_context_recall, 2),
                round(avg_context_recall, 2),
                round(median_context_recall, 2),
                round(std_context_recall, 2),
            ]
        )


def process_input_file(
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    stats_path: Path,
) -> None:
    """
    Load a single input JSON file, compute per-record results, write JSON output,
    and append aggregated metrics to the summary CSV.
    """
    raw = load_input(str(input_path))

    # ------------------------------------------------------------------
    # Extract hyperparameters from this input JSON if present
    # ------------------------------------------------------------------
    temperature = None
    topk = None
    similarity_threshold = None
    model = None

    if isinstance(raw, dict):
        temperature = raw.get("temperature")
        topk = raw.get("topK")
        similarity_threshold = raw.get("similarityThreshold")
        model = raw.get("model")

    # ------------------------------------------------------------------
    # Build results list from input format
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]]

    # Case A: { "records": [ {...}, {...}, ... ] }
    if isinstance(raw, dict) and "records" in raw and isinstance(raw["records"], list):
        results = [process_record(client, rec) for rec in raw["records"]]

    # Case B: single record object
    elif isinstance(raw, dict) and "query" in raw and "chunks" in raw:
        results = [process_record(client, raw)]

    else:
        raise ValueError(
            f"Unsupported input format in file {input_path}. Expected either:\n"
            '  { "records": [ {"query": ..., "chunks": ..., "chunkIds": ..., '
            '"chunkScores": ..., "reference_response": ..., "actual_response": ...}, ... ] }\n'
            "or a single record object with keys: query, chunks, chunkIds, chunkScores,\n"
            "reference_response, actual_response."
        )

    # ------------------------------------------------------------------
    # Save per-record JSON output
    # ------------------------------------------------------------------
    save_output(str(output_path), results)
    print(f"[OK] {input_path.name} → {output_path.name} ({len(results)} records)")

    # ------------------------------------------------------------------
    # Append aggregated statistics to CSV for THIS file
    # ------------------------------------------------------------------
    append_run_to_csv(
        input_path=str(input_path),
        results=results,
        csv_path=stats_path,
        temperature=temperature,
        topk=topk,
        similarity_threshold=similarity_threshold,
        model=model,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute relevant chunks and LLM-based metrics "
            "(faithfulness, factual_recall, factual_precision, context_recall) "
            "per query using a single LLM call per record.\n"
            "You can either:\n"
            "  - use --input/--output for a single file, OR\n"
            "  - use --inputdir (and optionally --outdir) to process all JSON files in a directory."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        required=False,
        help="Path to a single input JSON file (single-file mode)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=False,
        help="Path to output JSON file (single-file mode only)",
    )
    parser.add_argument(
        "--inputdir",
        required=False,
        help="Directory containing input JSON files (*.json) to be processed in batch mode",
    )
    parser.add_argument(
        "--outdir",
        required=False,
        help=(
            "Directory where output JSON files will be written in batch mode "
            "(requires --inputdir). If not provided, outputs are written next "
            "to the input files."
        ),
    )
    parser.add_argument(
        "--stat",
        "-s",
        required=False,
        default="./output/relevant_chunks_log.csv",
        help="Path to summary CSV file for aggregating metrics over runs",
    )

    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY in the environment.")

    # --outdir is only allowed with --inputdir
    if args.outdir and not args.inputdir:
        parser.error("--outdir can only be used together with --inputdir.")

    client = get_client()
    stats_path = Path(args.stat)
    date_suffix = datetime.now().strftime("%Y%m%d")

    # ------------------------------------------------------------------
    # Directory mode: --inputdir provided
    # ------------------------------------------------------------------
    if args.inputdir:
        dir_path = Path(args.inputdir)
        if not dir_path.is_dir():
            raise ValueError(
                f"--inputdir '{args.inputdir}' is not a directory or does not exist"
            )

        # Ignore --output in directory mode (as requested earlier)
        if args.output:
            print("Note: --output is ignored because --inputdir is provided.")

        json_files = sorted(
            [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
        )

        if not json_files:
            print(f"No .json files found in directory: {dir_path}")
            return

        # Determine base output directory in directory mode
        if args.outdir:
            output_base_dir = Path(args.outdir)
        else:
            # Default: write outputs next to input files
            output_base_dir = dir_path

        output_base_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {len(json_files)} JSON files in directory: {dir_path}")
        print(f"Outputs will be written to: {output_base_dir}")

        for input_path in json_files:
            output_name = f"{input_path.stem}_out{date_suffix}{input_path.suffix}"
            output_path = output_base_dir / output_name

            process_input_file(
                input_path=input_path,
                output_path=output_path,
                client=client,
                stats_path=stats_path,
            )

            print(f"  Sleeping {SLEEP_SECONDS} seconds to respect rate limits...")
            time.sleep(SLEEP_SECONDS)

        print(f"All files processed. Summary written to {stats_path}")
        return

    # ------------------------------------------------------------------
    # Single-file mode: require --input AND --output
    # ------------------------------------------------------------------
    if not args.input or not args.output:
        parser.error(
            "You must either:\n"
            "  - provide --inputdir (and optionally --outdir) to process a directory, OR\n"
            "  - provide BOTH --input and --output for a single file."
        )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise ValueError(f"--input '{args.input}' does not exist or is not a file")

    process_input_file(
        input_path=input_path,
        output_path=output_path,
        client=client,
        stats_path=stats_path,
    )

    print(f"Single file processed. Summary written to {stats_path}")


if __name__ == "__main__":
    main()
