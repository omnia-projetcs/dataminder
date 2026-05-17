import ollama

def summarize_text(text, model_name="gemma3:12b", level=7):
    if not text or not text.strip():
        return "No text provided for summarization."

    # Adjust instruction based on level
    if level < 1:
        level = 1
    elif level > 10:
        level = 10
        
    length_instruction = f"The desired detail level is {level} out of 10. "
    if level <= 2:
        length_instruction += "Make it extremely brief and concise. Only the most critical information."
    elif level <= 5:
        length_instruction += "Provide a balanced summary covering main points."
    elif level <= 8:
        length_instruction += "Provide a detailed and thorough extraction. Include all important technical details, examples, and specifics."
    else:
        length_instruction += "Be MAXIMALLY exhaustive. Extract EVERY piece of knowledge, every detail, every example, every command, every configuration. Do not omit anything. The output should be as long and complete as possible."

    # Prompt definition
    prompt = f"""You are a technical knowledge extractor. Your job is to read the following document and extract ALL concrete, useful knowledge from it.

{length_instruction}

RULES:
1. Write everything in ENGLISH regardless of the source language.
2. Extract ONLY real knowledge: technical facts, definitions, commands, tools, techniques, configurations, protocols, procedures, code examples, specific values, parameters, attack methods, defense strategies, etc.
3. Preserve code blocks, scripts, commands, and terminal output EXACTLY as they appear. Do not translate code.
4. Write the knowledge as standalone facts, as if you are a technical instructor teaching the subject.

FORBIDDEN — Do NOT include any of the following:
- Document metadata: title, author, publisher, ISBN, edition, publication date
- Structural descriptions: "Purpose and Philosophy", "Target Audience", "Document Structure", "Learning Objectives", "Prerequisites", "Overview of the book"
- Meta-references: "this book explains...", "the author describes...", "in chapter 3...", "according to the document..."
- Generic filler: "security is important", "cybersecurity is a growing field"
- Resources/References sections: URLs, links, contact information, email addresses, social media, "for more information visit...", recommended readings, bibliographies

Structure the output in Markdown using technical topic headings.

Document:
---
{text}
---
"""

    try:
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ], keep_alive=-1)
        return response['message']['content']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return f"Error during summarization: {e}\n\nCheck that the Ollama server is running and the model '{model_name}' is available."
