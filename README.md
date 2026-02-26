# Machine_Maintenance_Knowledge_Assistant
An AI powered knowledge assistant for machine maintenance. Uses machine manuals.
Uses Retrieval-Augmented Generation system that ingests documents (PDFs, manuals) and lets you query them.
Run locally with Ollama. Speed of inference depends on the PC hardware.e

## Prerequiste
```bash
1) pip install langchain langchain-chroma chromadb langchain-ollama streamlit python-dotenv pypdf

2) ollama pull llama3.1:8b
   ollama pull nomic-embed-text:latest
```
### The Assistant
1) E.g. asked Assistant on the avaliability of diagnostic features. Using human query.
   
![knassist](https://github.com/user-attachments/assets/ee8aa024-b6a5-4749-8ee0-9df642d42e45)

2) E.g. asked Assistant to do fault explanation.

![knassist2](https://github.com/user-attachments/assets/e4c7f0fc-b595-46c2-8d2d-e1e955985ad3)

3) Background processes.

   ![knassist3](https://github.com/user-attachments/assets/21dbbdcc-b7ab-4226-a586-525758848ad0)

   
