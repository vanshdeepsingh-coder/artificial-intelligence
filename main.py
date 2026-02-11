from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) # This is a paid API/Model

response = llm.invoke("Explain LangChain in one sentence.")
print(response.content)