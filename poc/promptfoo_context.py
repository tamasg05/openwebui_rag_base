import datetime
import json
import os


LOG_FILE_NAME = "./output/promptfoo_vars.log"
LOG_FILE = os.path.join(os.path.dirname(__file__), LOG_FILE_NAME)

# Ensure parent directories exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_to_file(message: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} - {message}\n")

def get_var(var_name, prompt, other_vars):
    # Log all inputs for inspection
    log_to_file(
        f"entering get_var() | var_name: {var_name} | prompt: {prompt} | other_vars: {json.dumps(other_vars, ensure_ascii=False)}"
    )

    # The document chunks have to be retrieved from the vector store, related to the query (Promptfoo's query var)
    retrieved_docs = [
            "Opel is a car.",
            "The price of Opel Astra depends on the regional markets.",
            "The Opel Astra price is about 9M HUF in the 2024 price listings.",
            "Its luggage rack is 300 liters.",
            "The Gellert Hill is in Budapest."
        ]


    try:
        if var_name == 'context':
            value = {
                'output': "\n".join(retrieved_docs)
            }
            log_to_file(f"Returning for context: {json.dumps(value, ensure_ascii=False)}")
            return value

        # Default variable value
        value = {'output': 'Document A, Document B, Document C, ...'}
        log_to_file(f"Returning default: {json.dumps(value, ensure_ascii=False)}")
        return value

    except Exception as e:
        error_message = f"Error in get_var(): {e}"
        log_to_file(error_message)
        return {'error': error_message}
