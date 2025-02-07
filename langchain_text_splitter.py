from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader

# Define quote
quote = """Words are flowing out like endless rain into a paper cup,
they slither while they pass,
they slip away across the universe."""

chunk_size = 24
chunk_overlap = 10

# CharacterTextSplitter Example
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

docs = splitter.split_text(quote)
print("CharacterTextSplitter Output:", docs)
print("Chunk Lengths:", [len(doc) for doc in docs])

# RecursiveCharacterTextSplitter Example
splitter = RecursiveCharacterTextSplitter(
    separators=["\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

docs = splitter.split_text(quote)
print("\nRecursiveCharacterTextSplitter Output:", docs)
print("Chunk Lengths:", [len(doc) for doc in docs])

# Load HTML Document
loader = UnstructuredHTMLLoader("white_house_executive_order_nov_2023.html")
data = loader.load()

# Define chunking parameters
chunk_size = 300
chunk_overlap = 100

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["."]
)

docs = splitter.split_documents(data)
print("\nSplit HTML Document:", docs)
