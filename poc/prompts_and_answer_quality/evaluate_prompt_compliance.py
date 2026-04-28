import argparse
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


LLM_MODEL = "gpt-5.1-2025-11-13"
LLM_TEMPERATURE = 0.0

OUTPUT_FILE = "prompt_compliance_assessment.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess how well an LLM dialog follows a global system prompt "
            "and an optional local system prompt."
        ),
        epilog="""
Examples:

  python assess_prompt_compliance.py \\
    --global-prompt-file global_prompt.txt \\
    --local-system-prompt-file local_system_prompt.txt \\
    --qa-dialog-file qa_dialog.txt

  python assess_prompt_compliance.py \\
    --global-prompt-file global_prompt.txt \\
    --qa-dialog-file qa_dialog.txt

  python assess_prompt_compliance.py \\
    --global-prompt-file prompts/global_prompt.txt \\
    --local-system-prompt-file prompts/local_system_prompt.txt \\
    --qa-dialog-file dialogs/qa_dialog.txt \\
    --output-file results/assessment.json

On Windows PowerShell, use backticks for line continuation:

  python assess_prompt_compliance.py `
    --global-prompt-file global_prompt.txt `
    --local-system-prompt-file local_system_prompt.txt `
    --qa-dialog-file qa_dialog.txt `
    --output-file result.json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--global-prompt-file",
        required=True,
        help="Path to the file containing the global system prompt.",
    )

    parser.add_argument(
        "--local-system-prompt-file",
        required=False,
        default=None,
        help=(
            "Optional path to the file containing the local system prompt. "
            "If omitted or empty, only the global prompt is evaluated."
        ),
    )

    parser.add_argument(
        "--qa-dialog-file",
        required=True,
        help="Path to the file containing the dialog to evaluate.",
    )

    parser.add_argument(
        "--output-file",
        default=OUTPUT_FILE,
        help=f"Path to the output JSON file. Default: {OUTPUT_FILE}",
    )

    return parser.parse_args()


def read_required_text_file(file_path: Path) -> str:
    """
    Reads a required UTF-8 text file and returns its stripped content.
    Raises FileNotFoundError if the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    return file_path.read_text(encoding="utf-8").strip()


def read_optional_text_file(file_path: Path | None) -> str:
    """
    Reads an optional UTF-8 text file.

    If the path is None, returns an empty string.
    If the file exists but is empty or contains only whitespace, returns an empty string.
    """
    if file_path is None:
        return ""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Optional local system prompt file not found: {file_path}"
        )

    return file_path.read_text(encoding="utf-8").strip()


def build_evaluation_prompt(
    global_system_prompt: str,
    local_system_prompt: str,
    qa_dialog: str,
) -> str:
    """
    Builds the prompt sent to the evaluator LLM.
    """

    has_local_prompt = bool(local_system_prompt.strip())

    local_prompt_section = (
        f"""
LOCAL SYSTEM PROMPT:
\"\"\"
{local_system_prompt}
\"\"\"
"""
        if has_local_prompt
        else """
LOCAL SYSTEM PROMPT:
No local system prompt was provided.
"""
    )

    local_prompt_instruction = (
        """
Also propose an improved version of the local system prompt if the dialog reveals
non-compliance that could be reduced by making the local prompt clearer, stricter,
or more explicit.

If no improvement is needed:
- return exactly "No change proposed." in "improved_local_prompt"
- explain in "improved_local_prompt_explanation" why no change was necessary
"""
        if has_local_prompt
        else """
No local system prompt was provided. Therefore:
- "local_prompt_compliance" must explain that no local prompt was provided.
- "improved_local_prompt" must be an empty string.
- "improved_local_prompt_explanation" must be an empty string.
"""
    )

    return f"""
You are a strict evaluator of LLM behavior.

Your task is to assess how well the assistant answers in the provided dialog comply with:

1. The global system prompt
2. The local system prompt, if one was provided

You must evaluate the assistant answers in the dialog, not the user messages.

Use the following grading scale:

5 = Complete match. The assistant answers fully comply with the applicable system prompt or prompts.
4 = Good match. Mostly compliant, with only minor issues.
3 = Partial match. Some important requirements are followed, but there are noticeable omissions or inconsistencies.
2 = Poor match. Many important requirements are ignored or contradicted.
1 = Very poor match. The assistant answers largely fail to follow the applicable prompt or prompts.

You may assign integer values only: 1, 2, 3, 4, or 5.

In addition to evaluating compliance, propose improved prompt versions.

For the global prompt:
- Always return "improved_global_prompt".
- If the original global prompt is already sufficient and no improvement is needed, return exactly "No change proposed.".
- In "improved_global_prompt_explanation", briefly explain what changed.
- If nothing changed, explain why no change was necessary.

For the local prompt:
{local_prompt_instruction}

The improved prompts should be practical, clear, and directly aimed at reducing the observed non-compliances in the dialog.
Do not invent unrelated new requirements.
Do not remove important requirements from the original prompts.
Preserve the original intent of the prompts.

Return your answer as valid JSON only.

The JSON format must be:

{{
  "mark": 1,
  "judgement": "Short textual explanation of the assessment.",
  "global_prompt_compliance": "Short explanation.",
  "local_prompt_compliance": "Short explanation.",
  "main_issues": [
    "Issue 1",
    "Issue 2"
  ],
  "improved_global_prompt": "Improved version of the global system prompt. Use exactly 'No change proposed.' if no improvement is needed.",
  "improved_global_prompt_explanation": "Explanation of what changed compared to the original global prompt, or why no change was needed.",
  "improved_local_prompt": "Improved version of the local system prompt. Use exactly 'No change proposed.' if no improvement is needed. Empty string if no local prompt was provided.",
  "improved_local_prompt_explanation": "Explanation of what changed compared to the original local prompt, or why no change was needed. Empty string if no local prompt was provided."
}}

Important rules:
- Return JSON only.
- Do not wrap the JSON in Markdown.
- The "mark" field must be an integer from 1 to 5.
- If the global system prompt needs no improvement, "improved_global_prompt" must be exactly "No change proposed.".
- If the global system prompt needs no improvement, "improved_global_prompt_explanation" must explain why no change was necessary.
- If a local system prompt was provided and no improvement is needed, "improved_local_prompt" must be exactly "No change proposed.".
- If a local system prompt was provided and no improvement is needed, "improved_local_prompt_explanation" must explain why no change was necessary.
- If no local system prompt was provided, "improved_local_prompt" must be "".
- If no local system prompt was provided, "improved_local_prompt_explanation" must be "".

GLOBAL SYSTEM PROMPT:
\"\"\"
{global_system_prompt}
\"\"\"

{local_prompt_section}

DIALOG TO EVALUATE:
\"\"\"
{qa_dialog}
\"\"\"
""".strip()


def call_llm(evaluation_prompt: str) -> dict:
    """
    Calls the OpenAI LLM and parses the JSON response.
    """

    client = OpenAI()

    response = client.responses.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict and careful evaluator. "
                    "You must return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": evaluation_prompt,
            },
        ],
    )

    raw_text = response.output_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The LLM response was not valid JSON.\n\n"
            f"Raw response:\n{raw_text}"
        ) from exc


def validate_result(result: dict, has_local_prompt: bool) -> None:
    """
    Performs basic validation of the evaluator output.
    """

    required_keys = {
        "mark",
        "judgement",
        "global_prompt_compliance",
        "local_prompt_compliance",
        "main_issues",
        "improved_global_prompt",
        "improved_global_prompt_explanation",
        "improved_local_prompt",
        "improved_local_prompt_explanation",
    }

    missing_keys = required_keys - result.keys()
    if missing_keys:
        raise ValueError(f"Missing keys in LLM response: {missing_keys}")

    mark = result["mark"]
    if not isinstance(mark, int) or mark < 1 or mark > 5:
        raise ValueError(f"'mark' must be an integer between 1 and 5. Got: {mark}")

    if not isinstance(result["main_issues"], list):
        raise ValueError("'main_issues' must be a list.")

    string_keys = {
        "judgement",
        "global_prompt_compliance",
        "local_prompt_compliance",
        "improved_global_prompt",
        "improved_global_prompt_explanation",
        "improved_local_prompt",
        "improved_local_prompt_explanation",
    }

    for key in string_keys:
        if not isinstance(result[key], str):
            raise ValueError(f"'{key}' must be a string.")

    if not has_local_prompt:
        if result["improved_local_prompt"] != "":
            raise ValueError(
                'When no local prompt is provided, "improved_local_prompt" must be an empty string.'
            )

        if result["improved_local_prompt_explanation"] != "":
            raise ValueError(
                'When no local prompt is provided, "improved_local_prompt_explanation" must be an empty string.'
            )


def resolve_input_path(file_name_or_path: str | None) -> Path | None:
    """
    Resolves a user-provided file name or path.

    If None is provided, returns None.
    If only a file name is given, it is interpreted relative to the current
    working directory.
    """
    if file_name_or_path is None:
        return None

    return Path(file_name_or_path).expanduser().resolve()


def main() -> None:
    args = parse_args()

    global_prompt_path = resolve_input_path(args.global_prompt_file)
    local_prompt_path = resolve_input_path(args.local_system_prompt_file)
    qa_dialog_path = resolve_input_path(args.qa_dialog_file)
    output_path = resolve_input_path(args.output_file)

    global_system_prompt = read_required_text_file(global_prompt_path)
    local_system_prompt = read_optional_text_file(local_prompt_path)
    qa_dialog = read_required_text_file(qa_dialog_path)

    has_local_prompt = bool(local_system_prompt.strip())

    evaluation_prompt = build_evaluation_prompt(
        global_system_prompt=global_system_prompt,
        local_system_prompt=local_system_prompt,
        qa_dialog=qa_dialog,
    )

    result = call_llm(evaluation_prompt)
    validate_result(result, has_local_prompt=has_local_prompt)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Assessment completed.")
    print(f"Mark: {result['mark']}")
    print(f"Judgement: {result['judgement']}")
    print(f"Full result written to: {output_path}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    main()