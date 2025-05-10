# 🛒 Retail AI Agent

An intelligent assistant for querying retail transaction data using both SQL and custom vector search powered by Qdrant and Google Gemini. Supports a Gradio web interface for interactive use.

---

## 🚀 Features

- ✅ Natural language understanding of user queries
- ✅ Hybrid routing to:
  - SQL database (`SQLite`)
  - Vector database (`Qdrant`) using precomputed embeddings
- ✅ Supports structured filters (e.g., branch, mall)
- ✅ Gemini LLM integration (via LangChain)
- ✅ Gradio interface for browser-based chat
- ✅ Dockerized deployment with `docker-compose`

---

## 📂 Project Structure
retail_ai_project/
├── retail_ai_agent.py # Main logic with Gradio
├── prompts.py # Prompt templates for LLM
├── scaler.joblib # Pretrained numerical scaler
├── encoder.joblib # One-hot encoder for categorical features
├── transactions.db # SQLite retail data
├── local_qdrant/ # Persistent vector DB directory
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md



---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repo

```bash
git clone https://github.com/yourusername/retail-ai-agent.git
cd retail-ai-agent
✅ 2. Add your environment variables
Either create a .env file or edit the docker-compose.yml:

env
Copy
Edit
GOOGLE_API_KEY=your-gemini-api-key
DATABASE_URL=sqlite:///transactions.db
✅ 3. Run with Docker Compose
bash
Copy
Edit
docker-compose build
docker-compose up
Then open http://localhost:7860 in your browser.

🧠 Prompts (in prompts.py)
The assistant uses 4 modular prompts:

ROUTER_PROMPT: Decides between SQL, Vector, or both

SQL_PROMPT: Generates SQL code

VECTOR_PROMPT: Generates vector search profiles

FINAL_PROMPT: Crafts a final answer using context + history

📦 Dependencies
All dependencies are in requirements.txt and include:

gradio

pandas, numpy, sqlalchemy, joblib, scikit-learn

langchain, langchain-google-genai

qdrant-client

🧪 Example Queries
Try these on the Gradio UI:

"What were total sales in City Mall during February?"

"Find refund transactions from weekend evenings"

"Show failed transactions with high tax amounts"

💡 License
MIT License. Feel free to fork and extend.
