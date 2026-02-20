from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import os

# Importation de tes fichiers locaux
from rag_service import RAGService
from rag_config import RAGConfig

# 1. Initialisation du Service RAG (avant l'app)
# On utilise RAGConfig défini dans ton fichier config
rag = RAGService(RAGConfig)

# 2. Configuration FastAPI
app = FastAPI(
    title="RAG Assistant",
    description="Interface de recherche pour codebase",
    version="0.3"
)

templates = Jinja2Templates(directory="templates")

# 3. Modèles Pydantic
class QueryRequest(BaseModel): 
    question: str

# 4. Routes de l'interface
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stats")
async def get_stats():
    return rag.get_stats()

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        answer = rag.chain.invoke(request.question).strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-index")
async def run_index():
    count, logs = rag.run_indexing()
    if count == 0: 
        return {"status": "error", "message": logs[0], "debug": logs}
    return {"status": "success", "message": "Indexation terminée", "count": count, "debug": logs}

@app.post("/clear-rag")
async def clear_rag():
    success, message = rag.clear_all_rag_data()
    if not success:
        return {"status": "error", "message": message}
    return {"status": "success", "message": message}

@app.get("/viewer", response_class=HTMLResponse)
async def viewer(request: Request):
    stats = rag.get_stats()
    
    return templates.TemplateResponse("viewer.html", {
        "request": request, 
        "summary": stats 
    })

@app.get("/file-chunks")
async def get_file_chunks(path: str):
    chunks = rag.get_chunks_by_path(path)
    
    if not chunks:
        return {"chunks": [], "count": 0, "message": "Aucun segment trouvé pour ce chemin."}
        
    return {
        "chunks": chunks, 
        "count": len(chunks)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)