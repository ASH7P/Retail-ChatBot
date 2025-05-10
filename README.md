# 🛒 Retail AI Agent

An intelligent assistant for querying retail transaction data using both SQL and custom vector search powered by Qdrant and Google Gemini. Includes a clean Gradio-based chat interface and is fully Dockerized for deployment.

---

## 🚀 Features

- ✅ Understands natural language queries  
- ✅ Smart routing to:
  - SQL database (via SQLite)
  - Vector search (via Qdrant and your own engineered embeddings)  
- ✅ Supports filterable metadata (e.g. mall, branch, transaction status)  
- ✅ Integrates with Google Gemini via LangChain  
- ✅ Gradio-powered web interface  
- ✅ Fully containerized using Docker + Docker Compose

---

## 📂 Project Structure

```
retail_ai_project/
├── retail_ai_agent.py         # Main logic with Gradio
├── prompts.py                 # Prompt templates for LLM
├── scaler.joblib              # Pretrained scaler for numerical inputs
├── encoder.joblib             # One-hot encoder for categorical variables
├── transactions.db            # Local SQLite database
├── local_qdrant/              # Persistent Qdrant vector storage
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```
---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/yourusername/retail-ai-agent.git
cd retail-ai-agent
```

---

### ✅ 2. Add The API Key

To keep our API key secure, we did **not** hardcode it. Instead:

1. Create a `.env` file locally with the following contents:

    ```
    GOOGLE_API_KEY=the-key-from-google-drive
    DATABASE_URL=sqlite:///transactions.db
    ```

2. Upload the `.env` file to a secure Google Drive link.  
3. Share the link with trusted team members (view-only).  
4. Download the `.env` file into the project root **before running Docker**.

---

### ✅ 3. Build and Run the App

```bash
docker-compose build
docker-compose up
```

Then open your browser and visit: http://localhost:7860

---

## 🧠 Prompt Design (in `prompts.py`)

The assistant uses four modular prompts:

- `ROUTER_PROMPT`: Chooses between SQL or Vector routes  
- `SQL_PROMPT`: Generates valid SQL code  
- `VECTOR_PROMPT`: Creates vector search profiles  
- `FINAL_PROMPT`: Composes a final response using all data + history  

---

## 📦 Dependencies

Listed in `requirements.txt`:

- `gradio`
- `pandas`, `numpy`, `sqlalchemy`, `joblib`, `scikit-learn`
- `langchain`, `langchain-google-genai`
- `qdrant-client`

---

## 🧪 Example Queries

Try these in the Gradio interface:

- "What were total sales in C Mall during February?"
- "Find refund transactions from weekend evenings"
- "Show failed transactions with high tax amounts"

---

## 📄 License

MIT License — free to use, fork, and customize.
