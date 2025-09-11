1. Create an .env file in the project root directory with your api keys, take extra care that no spaces will be put after the last character of the keys
    e.g.
    OPENAI_API_KEY=sk-...                 # from https://platform.openai.com
    GOOGLE_API_KEY=AIza...                # from https://aistudio.google.com (Gemini)

2. Start up OpenWeb UI and LiteLLM (if you use windows install docker desktop at first)
    Execute the following command in the project's root folder:
        docker compose up -d
        this will download OpenWebUI and LiteLLM. The latter is set up as a proxy for non-OpenAI models.

3. Open http://localhost:3000
    create an admin account and log in
        create further users and user groups as you wish

4. Add OpenAI connection
    Settings (in the bottom left corner) → Admin Panel → Settings → Connections → OpenAI → Add Connection
    Go to the section OpenAI API
        Add the following URL and your OpenAI api key
            https://api.openai.com/v1
            test the connection

5. Check whether LiteLLM can access the Gemini Models listed in the yaml file:
    curl -X POST http://localhost:4000/v1/embeddings -H "Authorization: Bearer sk-local-my-supersecret-key" -H "Content-Type: application/json" -d "{\"model\":\"text-embedding-004\",\"input\":\"hello world\"}"
    If it works, then you will receive a vector, if it does not, then check the api keys, check whether LiteLLM is up and running, check the logs from LiteLLM
        to check what exactly is running:
            docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

        to check the GOOGLE_API_KEY environment variable:
            docker exec -it litellm printenv | findstr GOOGLE_API_KEY

        to check the LiteLLM logs:
            docker logs -n 50 litellm

        to test whether the models are listed for which LiteLLM works as a proxy
            curl -X GET http://localhost:4000/v1/models -H "Authorization: Bearer sk-local-my-supersecret-key"

6. Add LiteLLM as a connection 
    Settings (in the bottom left corner) → Admin Panel → Settings → Connections → OpenAI → Add Connection
    Go to the section OpenAI API
        Add the following URL and your LiteLLM api key you specified in docker-compose.yaml
            http://litellm:4000
            test the connection

7. Go to the Open Web UI front page, select the model you wish to use in the top left combo-box and ask a question to see everything works. You can only use the models in the free-tier, if you do not upload a certain sum of money for your api keys.

8. Set up RAG
    The document chunks will be written in a chroma db and persisted on the volume specified in docker-compose.yaml
     Settings (in the bottom left corner) → Admin Settings → Documents
        In the Embedding Model Engine section, select OpenAI (also for non-OpenAI models)
            If you want to use a non-OpenAI model, then
                (1) set the model url to LiteLLM: http://litellm:4000
                (2) set the api key for LiteLLM specified in the docker-compose.yaml: sk-local-my-supersecret-key
                (3) specify an embedding model, e.g. text-embedding-004 from Google
        Specify the chunk length and overlap
        Specify how many chunks shall be returned and added to the context
        Specify the RAG prompt, a template is also provided

    Then go to the Open Web UI front page
        Workspace → Knowledge → Create a knowledge base → set access rights and add a brief description
            Add documents to your collection (files, directories)

    Go to the front page, select New Chat 
        In the chat window, you can reference your collection with a hash-tag e.g. #test means the collection named test, and you can ask questions that will be answered based on the collection by the model you selected.



