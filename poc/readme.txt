The proof of concept had the following goals:
-to see which tool is the most handy to automatically test different RAG metrics;
-reproducibility: the tests were performed with a fixed set of document chunks to see whether the metrics change if the same tests are repeated, i.e. the chunks do not come from vector stores. Nevertheless, demo_query_to_openwebuicollection.py demonstrates how to connect an existing OpenWeb UI application through an API key from where the RAG chunks could also be retrieved.

Setting up the environment:
1.a) Creating a virtual environment:
    cd poc
    python -m venv ragtest_venv

1.b) Activating the virtual environment
    . ./ragtest_venv/Scripts/activate
    You need to activate this environment each time you use the tools

2.) Installing necessary libraries:
    pip install -r requirements.txt 

3.) If you want to use an OpenWeb UI RAG collection:
    a.)  set an api-key for OpenWeb UI, so that the demo_query_to_openwebuicollection.py scripts can connect OpenWeb UI (you can generate such a key in Open Web UI).
    
    b.) set the collection ID for the RAG collection in OpenWeb UI:
        -open the collection in the OpenWeb UI and copy the ID:
            e.g. http://localhost:3000/workspace/knowledge/1ef62902-82c1-4ebb-b0d8-bef2f2aa93a9
                  then 1ef62902-82c1-4ebb-b0d8-bef2f2aa93a9

5.) Setting the api-keys for the LLMs (.env not checked in; it contains OPENAI_API_KEY and GOOGLE_API_KEY)

6.) Running the DeepEval application:
    pytest -v -s (-s for being able to see the print statements)
    It will produce an output table in csv in ./output/deepeval_metrics.csv. If the directory does not exist, the script will create it.

7.) Running the RAGAS application
    It is a plain python app, you can run it from the command line or IDE, it wil produce an output table in json in ./output/ragas_per_row_metrics.json and reads the input from ./input/demo_data.json. If the output directory does not exist, the script will create it. The csv output type was changed to json as the LLM can produce several-line answers with commas.

Take care as the different tools can compute the metrics: recall, precision, faithfulness and factual_correctness in a different manner. See the comments added to the code.


8.) Promptfoo:
        a.) Install Node.js 20 or newer
        b.) npm install -g promptfoo
        c.) promptfoo init
        d.) set the OPENAI_API_KEY and GOOGLE_API_KEY environment variables if they are not yet set in .env
        e.) edit promptfooconfig.yaml
        f.) promptfoo eval (to run the tests)
            if you want to see more output:
                LOG_LEVEL=debug promptfoo eval
        g.) promptfoo view (to generate the report)

DeepEval:
    -Very strict by default: Opel Astra vs. Open Astra in the retrieved chunks and in the actual answer: faithfulness: 0
    -Can give a reason for each metric's judgement
    -Precision: ranking based

RAGAS:
    -Considers typos in an appropriate manner: Opel Astra vs. Open Astra in the retrieved chunks and in the actual answer: faithfulness: 1.0
    -Context relevance: a custom metric needs to be set up if necessary
                        it is demonstrated how to define and use a custom metric that computes the context relevance.
    -Precision: ranking based

Promptfoo:
    -Several models and prompts can be evaluated in 1 configuration in one go.
    -Precision: not available by default
    -Context relevance: Not ranking based, it shows the ratio of the useful and the retrieved data in the context.
    -Context Recall: contradiction with DeepEval and RAGAS results using the same LLM
        promptfoo context recall:
            score: 0.11
            expected answer: "The Opel Astra price is about 9M HUF in the 2024 price listings."
            actual answer: "About 9 million HUF in the 2024 price listings, though actual prices can vary by regional market."
            Retrieved context: 
                        "Opel is a car.",
                        "The price of Opel Astra depends on the regional markets.",
                        "The Opel Astra price is about 9M HUF in the 2024 price listings.",
                        "Its luggage rack is 300 liters.",
                        "The Gellert Hill is in Budapest."
            Known bug: https://github.com/promptfoo/promptfoo/issues/1506
                    
    -Possibility to make new metrics very easily
        -As a demonstration, see the workaround for the context recall metric and named it as context_recall_wa

In conclusion, we need a tool that can evaluate the output of our software after the data processing takes place in the software under evaluation. From this respect, RAGAS would qualify as we can make it evaluate the software's output and create the quality metrics. Nevertheless, we experienced several difficulties with RAGAS including: (1) high-token consumption by the LLM, resulting in higher costs, (2) rate limiters at the LLM's side produce exceptions due to the heavy load. For these reasons the compute_relevant_chunks.py script was developed that computes all the necessary metrics including the number of the relevant chunks, using fuzzy sets, i.e. a membership value is computed for the given metrics. 

The new script cut the costs to approx. 5% compared to the RAGAS script's costs and helped to avoid the rate limiter exceptions. In addition the number of relevant chunks are computed with a threshold value set in the script for classifying the relevance scores which represent a fuzzy membership value. The compute_relevant_chunks.py script's expected input is documented in the script. For each input json file an output json file is produced and a line in the summary CSV file with the metrics. The script's switches are also documented and a help is available with the --help switch.

The script create_charts2.py creates the visual representations of the metrics assuming the test runs happen with different temperature and topk combinations in the software under evaluation. The script compute_relevant_chunks.py uses temperature=0 for decreasing the stochastic nature of the LLM responses.
