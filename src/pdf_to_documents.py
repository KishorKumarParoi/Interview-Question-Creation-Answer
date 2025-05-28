import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from src.prompt import refine_template, prompt_template

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def file_processing(file_path="./data/SDG.pdf"):
    """
    Loads a PDF, splits its content for question and answer generation, and returns:
    - documents_ques_gen: List of Document objects for question generation
    - document_ans_gen: List of Document objects for answer generation
    """

    # Load PDF data
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Combine all page content into a single string
    ques_gen = ""
    for page in data:
        ques_gen += page.page_content

    # Split the text into chunks for question generation
    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo", chunk_size=10000, chunk_overlap=200
    )

    chunk_ques_gen = splitter_ques_gen.split_text(ques_gen)

    # Convert each chunk to a Document for question generation
    documents_ques_gen = [
        Document(page_content=chunk, metadata={"chunk_number": i + 1})
        for i, chunk in enumerate(chunk_ques_gen)
    ]

    # Split the documents further for answer generation
    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo", chunk_size=1000, chunk_overlap=100
    )

    document_ans_gen = splitter_ans_gen.split_documents(documents_ques_gen)

    return documents_ques_gen, document_ans_gen


def llm_pipeline(file_path):
    llm_ques_gen_pipeline = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template, input_variables=["text"]
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"], template=refine_template
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    ques = ques_gen_chain.run(documents_ques_gen)

    embeddings = OpenAIEmbeddings()
    llm_ans_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    vector_store = FAISS.from_documents(document_ans_gen, embeddings)
    ques_list = ques.split("\n")
    answer_gen_chain = RetrievalQA.from_chain_type(
        llm=llm_ans_gen, chain_type="stuff", retriever=vector_store.as_retriever()
    )

    return ques_list, answer_gen_chain


file_path = "./data/SDG.pdf"
documents_ques_gen, document_ans_gen = file_processing(file_path)
ques_list, answer_gen_chain = llm_pipeline(file_path)


print(document_ans_gen, documents_ques_gen, ques_list, answer_gen_chain)
