from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.sql import SQLDatabaseChain
from langchain_classic.chains.sql_database.prompt import PROMPT_SUFFIX

from few_shots import few_shots
from prompts import MYSQL_PROMPT

import os
from dotenv import load_dotenv

load_dotenv()


def get_few_shot_db_chain():
    db_user = "root"
    db_password = ""
    db_host = "localhost"
    db_name = "tshirts"

    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3,
    )
    llm = GoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
        temperature=0.1,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    example_prompt = PromptTemplate(
        input_variables=[
            "Question",
            "SQLQuery",
            "SQLResult",
            "Answer",
        ],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=MYSQL_PROMPT,
        suffix=PROMPT_SUFFIX,
        input_variables=[
            "input",
            "table_info",
            "top_k",
        ],  # These variables are used in the prefix and suffix
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain


if __name__ == "__main__":
    chain = get_few_shot_db_chain()
    response = chain.run("what is the total number of T-shirts available?")
    print("response: ", response)
