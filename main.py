import os
from extractor import extract_text
from summarizer import summarize_text

# Configuration
INPUT_DIR = "input_docs"
OUTPUT_DIR = "output_summaries"
OLLAMA_MODEL = "llama3" # You can change this to mistral, mixtral, phi3, etc.

def setup_directories():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Directories configured: '{INPUT_DIR}/' (for your documents) and '{OUTPUT_DIR}/' (for the summaries).")

def process_documents():
    setup_directories()
    
    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
    
    if not files:
        print(f"No documents found in the '{INPUT_DIR}/' directory. Please place your files (txt, md, pdf, docx) there.")
        return
        
    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.md")
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"[{filename}] Summary already exists. Skipping to the next file...")
            continue
            
        print(f"[{filename}] 1/3 Extracting text...")
        text = extract_text(input_path)
        
        if not text.strip():
            print(f"[{filename}] No text could be extracted.")
            continue
            
        print(f"[{filename}] 2/3 Generating summary via Ollama (model {OLLAMA_MODEL}). This may take a few moments...")
        summary_md = summarize_text(text, model_name=OLLAMA_MODEL)
        
        print(f"[{filename}] 3/3 Saving summary...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
            
        print(f"[{filename}] Successfully completed! (Saved in {output_path})\n")

if __name__ == "__main__":
    print("--- Starting document processing ---")
    process_documents()
