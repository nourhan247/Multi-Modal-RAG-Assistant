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

# def load_documents(docs_path='data'):
    
#     if not os.path.exists(docs_path):
#         raise FileNotFoundError("Directory not found")
    
    
#     load_doc=DirectoryLoader(
#         path=docs_path,
#         glob='*.pdf',
#         loader_cls=PyPDFLoader
#     )
    
#     documents=load_doc.load()
    
#     if len(documents)==0:
#             raise FileNotFoundError("No files in this directory")
        
#     for i,doc in enumerate(documents[:3]):
#         print(f"\nDocument {i+1}:")
#         print(f" source {doc.metadata['source']}")
#         print(f" metadata {doc.metadata}")
#         print(f" content length {len(doc.page_content)}")
#         print(f" content preview {doc.page_content[:100]}")
    
#     return documents


def partition_documents(docs_path="data"):
    
    pdfs=[]
    for file_path in Path(docs_path).glob("*.pdf"):
        
        pdf=partition_pdf(
            filename=file_path,
            strategy='hi_res',
            extract_image_block_to_payload=True,
            extract_image_block_types=['Image'],
            infer_table_structure=True
        )
        pdfs.extend(pdf)
         
        
    if not pdfs:
        raise FileNotFoundError("No PDF files found in this directory")  
    
    print(f"\nTotal extracted elements: {len(pdfs)}")

    for i, element in enumerate(pdfs[:3]):
        print(f"\nElement {i + 1}")
        print("Type:", type(element).__name__)
        print("Text length:", len(str(element)))
        print("Text preview:", str(element)[:300])
        print("Metadata:", element.metadata)  
        
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
    
    
def summarize_chunks(chunks):
    
    print("    Summarize Chunks   ")
    
    langchain_documents=[]
    
    total_chunks=len(chunks)
    
    for i,chunk in enumerate(chunks):
        content_type=separate_content_types(chunk)
    
        if content_type['table'] or content_type['image']:
            try:
                enhanced_summary=ai_summarization(
                    content_type['text'],
                    content_type['table'],
                    content_type['image']
                )
                print(f"     → AI summary created successfully")
                print(f"     → Enhanced content preview: {enhanced_summary[:200]}...")
                
            except Exception as e:
                print(f"   AI Summarization Failed: {e}    ")
                enhanced_summary=content_type['text']
                
        else:
            print(f"     → Using raw text (no tables/images)")
            enhanced_summary = content_type['text']   
                
        doc=Document(
            page_content=enhanced_summary,
            metadata={
                    "original_content": json.dumps({
                        "raw_text": content_type['text'],
                        "tables_html": content_type['table'],
                        "images_base64": content_type['image']
                    })
                }
            )  
        langchain_documents.append(doc)
            
    print(f" Processed {len(langchain_documents)} chunks")  
        
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

def create_vectorstore(chunks,persist_directory='dbv2/chroma_db'):
    embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(" Finished creating vector store")
    print(f"Vector store created and saved to {persist_directory}")
    
    return vector_store
    
    
def main():
    """Main ingestion pipeline."""
    print("=== RAG Document Ingestion Pipeline ===\n")

    docs_path = "data"
    json_file='chunks.json'
    persistent_directory = "dbv2/chroma_db"
    
    database_file = os.path.join(
        persistent_directory, "chroma.sqlite3"
    )

 
    if os.path.isfile(database_file):
        print("Checking existing vector store...")

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        count = vectorstore._collection.count()
        print(f"Existing vectors: {count}")

        if count > 0:
            print("Loading existing vector store...")
            return vectorstore

        print("Vector store is empty. Rebuilding it...")
        
    # documents=load_documents(docs_path)    
    pdfs=partition_documents(docs_path)   
    chunks=create_chunks(pdfs) 
    
    print(f"\nTotal chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1}")
        print("Type:", type(chunk).__name__)
        print("Text length:", len(chunk.text))
        print("Text preview:\n", chunk.text[:500])
        print("Metadata:", chunk.metadata)
        
    summarized_chunks=summarize_chunks(chunks)
    

    vectorstore = create_vectorstore(
        summarized_chunks,
        persistent_directory
    )

    print(
        "\nIngestion complete! "
        "Your documents are ready for RAG queries."
    )

    return vectorstore


if __name__ == "__main__":
    main()