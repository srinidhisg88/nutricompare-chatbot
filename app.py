from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import re
import os

load_dotenv()

app = FastAPI()

# Load environment variables
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Request schema
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Chatbot is running!"}

# Utility functions
def configure_db():
    try:
        conn_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        engine = create_engine(conn_string)
        return SQLDatabase(engine), engine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def extract_sql_query(response):
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()

    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
    for keyword in sql_keywords:
        match = re.search(f'{keyword}\\s+.*', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()

    return None

def is_valid_sql(query):
    return any(kw in query.upper() for kw in ['SELECT', 'FROM', 'WHERE'])

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile"
        )

        db, engine = configure_db()

        # 1. Attempt SQL query generation
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

        sql_prompt = f"""
        Based on this user question, return a valid SQL query only if it relates to the juice database.

        User: "{request.query}"

        Only return the SQL query (no explanation). If unrelated to the database, respond with: "NON_DB".
        """

        agent_response = agent.run(sql_prompt)

        if "NON_DB" in agent_response:
            # 2. Handle non-database queries
            general_answer = llm.invoke(request.query).content
            return {
                "query": request.query,
                "sql_query": None,
                "summary": general_answer,
                "source": "LLM (no DB)"
            }

        # 3. Parse and validate SQL
        sql_query = extract_sql_query(agent_response)
        if not sql_query or not is_valid_sql(sql_query):
            raise ValueError("Failed to extract a valid SQL query.")

        # 4. Execute SQL and fetch result
        result_list = []
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            if result.returns_rows:
                cols = result.keys()
                for row in result:
                    row_dict = {col: (str(val) if not isinstance(val, (str, int, float, bool, type(None))) else val) for col, val in zip(cols, row)}
                    result_list.append(row_dict)
                result_str = json.dumps(result_list)
            else:
                result_str = "Query executed successfully. No rows returned."

        # 5. Summarize with personalization
        summary_prompt = f"""
        The user asked: "{request.query}"
        The SQL query was: {sql_query}
        The result from the database is: {result_str}

        Provide a friendly, human-like explanation of the results.
        If possible, suggest healthier or more personalized juice options based on the result.
        """
        summary = llm.invoke(summary_prompt).content

        return {
            "query": request.query,
            "sql_query": sql_query,
            "sql_result": result_list,
            "summary": summary,
            "source": "DB + LLM"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")