# Directory loader is used to load all the documents from a directory.
# It can be used to load multiple documents from a directory 
# We can enter the pattern of the files we want to load. For example, if we want to load all the text files from a directory, we can use the pattern "*.txt".

# We will use LAZY_LOAD instead of .load to load the documents. LAZY_LOAD will load the documents one by one when we access them, instead of loading all the documents at once. This is useful when we have a large number of documents and we don't want to load them all into memory at once.

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader("docs/", 
                         glob="*.pdf",  # Define pattern to match PDF files
                         show_progress=True,
                         loader_cls= PyPDFLoader) # Specify the loader class to use for loading PDF files

documents = list(loader.lazy_load())
# print("Number of documents loaded:", len(documents)) # We will find that the number of documents loaded is equal to the number of PDF files in the directory.
# print("First document content:", documents[0].page_content)
print("First document metadata:", documents[0].metadata)
