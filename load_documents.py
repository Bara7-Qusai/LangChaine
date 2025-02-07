# Import necessary libraries
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, UnstructuredHTMLLoader

# Load CSV file
csv_loader = CSVLoader('fifa_countries_audience.csv')
csv_data = csv_loader.load()
print("CSV Data:", csv_data[0])

# Load PDF file
pdf_loader = PyPDFLoader("rag_vs_fine_tuning.pdf")
pdf_data = pdf_loader.load()
print("PDF Data:", pdf_data[0])

# Load HTML file
html_loader = UnstructuredHTMLLoader('white_house_executive_order_nov_2023.html')
html_data = html_loader.load()
print("HTML Data:", html_data[0])

# Print metadata for the HTML document
print("HTML Metadata:", html_data[0].metadata)
