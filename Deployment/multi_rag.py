import os
# from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from pathlib import Path
from typing import List
import json

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

load_dotenv()



def partition_documents(pdf_path):
    pdf_path = Path(pdf_path)

    pdfs = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        extract_image_block_to_payload=True,
        extract_image_block_types=["Image"],
        infer_table_structure=True
    )

    if not pdfs:
        raise ValueError(f"No elements extracted from {pdf_path.name}")

    print(f"\nPDF: {pdf_path.name}")
    print(f"Extracted elements: {len(pdfs)}")

    return pdfs


def create_chunks(pdfs):

    chunks=chunk_by_title(
            elements=pdfs,
            combine_text_under_n_chars=500,
            max_characters=3000,
            new_after_n_chars=2400
            
        )
    return chunks 

# elements = partition_documents("data")
# chunks = create_chunks(elements)

# print("Extracted elements:", len(elements))
# print("Created chunks:", len(chunks))

# if chunks:
#     print(chunks[0].text[:500])

def separate_content_types(chunks):
    content_data={
        'text':chunks.text,
        'image':[],
        'table':[],
        'type':['text']
    }
    if hasattr(chunks,'metadata') and hasattr(chunks.metadata,'orig_elements'):
        for element in chunks.metadata.orig_elements:
            element_type=type(element).__name__
            
            if element_type == 'Table':
                content_data['type'].append('table')
                table_html=getattr(element.metadata,'text_as_html',element.text)
                content_data['table'].append(table_html)
                
            elif element_type=='Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['type'].append('image')
                    content_data['image'].append(element.metadata.image_base64)
    
    content_data['type'] = list(set(content_data['type']))
    
    print("\n--- Content Type Analysis ---")
    print("Text length:", len(content_data["text"]))
    print("Tables found:", len(content_data["table"]))
    print("Images found:", len(content_data["image"]))
    print("Detected types:", content_data["type"])
    
    return content_data

def ai_summarization(text:str,tables:List[str],images:List[str]):
    print("    AI Summarization    ")
    try:
        
        llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0
        )

        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""
        message_content = [{"type": "text", "text": prompt_text}]
        
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"      AI summary failed: {e}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary
    
    
def summarize_chunks(chunks, source):
    langchain_documents = []

    for chunk in chunks:
        content_type = separate_content_types(chunk)

        if content_type["table"] or content_type["image"]:
            enhanced_summary = ai_summarization(
                content_type["text"],
                content_type["table"],
                content_type["image"]
            )
        else:
            enhanced_summary = content_type["text"]

        doc = Document(
            page_content=enhanced_summary,
            metadata={
                "source": source,
                "original_content": json.dumps({
                    "raw_text": content_type["text"],
                    "tables_html": content_type["table"],
                    "images_base64": content_type["image"]
                })
            }
        )

        langchain_documents.append(doc)

    print(f"Processed {len(langchain_documents)} chunks from {source}")
    return langchain_documents 

def export_chunks_json(chunks,filename='chunks.json'):
    export_data=[]
    for i,doc in enumerate(chunks):
        chunk={
            'chunk_id':1+i,
            'content':doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk)
        
    with open(filename,'w', encoding='utf-8') as f:
        json.dump(export_data,f, indent=2, ensure_ascii=False)    
        
    print(f" Exported {len(export_data)} chunks to {filename}")  
      
    return export_data

import hashlib

def create_vectorstore(
    chunks,
    source_path,
    persist_directory="dbv2/chroma_db"
):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db_path = Path(persist_directory).resolve()
    db_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=str(db_path),
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    file_hash = hashlib.sha256(
        Path(source_path).read_bytes()
    ).hexdigest()

    ids = [
        hashlib.sha256(f"{file_hash}::{i}".encode()).hexdigest()
        for i in range(len(chunks))
    ]

    before = vectorstore._collection.count()

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    after = vectorstore._collection.count()

    print(f"Vectors before: {before}")
    print(f"Vectors after:  {after}")
    print(f"New vectors:    {after - before}")

    return vectorstore
    
    
def main(pdf_paths=None):
    print("=== Incremental RAG Ingestion ===")

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    db_dir = base_dir / "dbv2" / "chroma_db"

    if pdf_paths is None:
        pdf_paths = list(data_dir.glob("*.pdf"))
    else:
        pdf_paths = [Path(p).resolve() for p in pdf_paths]

    if not pdf_paths:
        raise FileNotFoundError("No PDF files to process.")

    vectorstore = None

    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path.name}")

        elements = partition_documents(pdf_path)
        chunks = create_chunks(elements)

        print(f"Created chunks: {len(chunks)}")

        summarized_chunks = summarize_chunks(
            chunks,
            source=pdf_path.name
        )

        vectorstore = create_vectorstore(
            summarized_chunks,
            source_path=pdf_path,
            persist_directory=str(db_dir)
        )

    print("\nIncremental ingestion complete!")
    return vectorstore


if __name__ == "__main__":
    main()