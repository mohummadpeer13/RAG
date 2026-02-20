import os
import shutil
import glob
import uuid
import time
import gc
from pathlib import Path

# Imports LangChain & Chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class RAGService:
    def __init__(self, config):
        # Si tu es dans Docker, host.docker.internal pointe vers ta machine
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        """Initialise les modèles et détecte la base existante."""
        self.config = config
        self.embeddings = OllamaEmbeddings(
            model=config['EMBEDDING_MODEL'],
            base_url=ollama_host
        )

        self.llm = ChatOllama(
            model=config['MODEL_NAME'],
            temperature=0.25,
            base_url=ollama_host
        )
        
        # Détection du dossier le plus récent au démarrage
        self.persist_dir = self._get_latest_dir()
        self.vector_db = None
        self.retriever = None
        self.chain = None
        
        # Chargement initial
        self.initialize_rag()

    def _get_latest_dir(self):
        """Trouve le dossier chroma_db le plus récent sur le disque."""
        dirs = glob.glob("./chroma_db*")
        if not dirs:
            return "./chroma_db"
        return max(dirs, key=os.path.getmtime)

    def initialize_rag(self):
        """Configure (ou reconfigure) la base de vecteurs et la chaîne de réponse."""
        if os.path.exists(self.persist_dir):
            self.vector_db = Chroma(
                persist_directory=self.persist_dir, 
                embedding_function=self.embeddings
            )
        else:
            # Création d'une instance vide si aucun dossier n'existe
            self.vector_db = Chroma(
                persist_directory="./chroma_db", 
                embedding_function=self.embeddings
            )
        
        # Mise à jour du retriever et de la chain
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": self.config.get('RETRIEVER_K', 5)})
        prompt = ChatPromptTemplate.from_template(self.config['TEMPLATE'])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Puis modifie ta chaine dans initialize_rag :
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt 
            | self.llm 
            | StrOutputParser()
        )

    def run_indexing(self):
        """Exécute le processus complet d'indexation avec normalisation des chemins."""
        debug_log = []
        try:
            # 1. Libération des ressources
            debug_log.append("Nettoyage de la mémoire...")
            self.vector_db = None
            self.retriever = None
            self.chain = None
            gc.collect()
            time.sleep(1.5)

            # 2. Chargement des fichiers
            docs = []
            for ext in ["**/*.java", "**/*.xml", "**/*.properties"]:
                loader = DirectoryLoader(self.config['DOC_PATH'], glob=ext, silent_errors=True)
                docs.extend(loader.load())
            
            debug_log.append(f"{len(docs)} fichiers chargés.")

            # 3. Découpage en chunks spécialisé Java
            # On utilise le splitter configuré pour le langage JAVA
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.JAVA,
                chunk_size=self.config['CHUNK_SIZE'], 
                chunk_overlap=self.config['CHUNK_OVERLAP']
            )
            chunks = text_splitter.split_documents(docs)
            debug_log.append(f"{len(chunks)} segments créés avec le parser JAVA.")

            # --- NORMALISATION DES MÉTADONNÉES ---
            for chunk in chunks:
                raw_path = chunk.metadata.get("source", "")
                # 1. On remplace les \ par /
                # 2. On s'assure que le chemin est relatif pour éviter les chemins C:/Users/...
                clean_path = raw_path.replace("\\", "/")
                chunk.metadata["source"] = clean_path
            # --------------------------------------

            # 4. Création du dossier unique
            unique_id = str(uuid.uuid4())[:8]
            new_dir = f"./chroma_db_{unique_id}"
            debug_log.append(f"Stockage dans : {new_dir}")

            # 5. Création de la nouvelle base
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=new_dir
            )

            # 6. Rechargement du moteur
            self.persist_dir = new_dir
            self.initialize_rag()

            # 7. Ménage des anciens dossiers
            for folder in glob.glob("./chroma_db_*"):
                if folder != new_dir:
                    shutil.rmtree(folder, ignore_errors=True)

            return len(chunks), debug_log

        except Exception as e:
            return 0, [f"Erreur d'indexation : {str(e)}"]

    def get_stats(self):
        try:
            # On récupère les données pour calculer la taille moyenne
            all_data = self.vector_db.get(include=["metadatas", "documents"])
            ids = all_data.get("ids", [])
            metas = all_data.get("metadatas", [])
            docs = all_data.get("documents", [])
            
            total_chunks = len(ids)
            
            # Calcul de la taille moyenne des textes
            avg_size = sum(len(d) for d in docs) / total_chunks if total_chunks > 0 else 0
            
            # Extraction des sources uniques
            sources = sorted(list(set(m.get("source", "Inconnu") for m in metas)))
            
            # Calcul des packages (basé sur le dossier des fichiers java)
            unique_pkgs = set()
            for s in sources:
                if "/" in s:
                    pkg = "/".join(s.split("/")[:-1])
                    unique_pkgs.add(pkg)

            # Retourne TOUTES les clés attendues par le JS de l'index.html
            return {
                "total_chunks": total_chunks,
                "average_chunk_size": int(avg_size),
                "total_files": len(sources),
                "unique_packages": len(unique_pkgs),
                "embedding_model": self.config.get('EMBEDDING_MODEL'),
                "llm_model": self.config.get('MODEL_NAME'),
                "persist_dir": self.persist_dir,
                "chunk_size": self.config.get('CHUNK_SIZE'),
                "chunk_overlap": self.config.get('CHUNK_OVERLAP'),
                "retriever_k": self.config.get('RETRIEVER_K', 5),
                "files": sources # Pour le viewer
            }
        except Exception as e:
            print(f"Erreur stats: {e}")
            return {"error": str(e)}
        
    def clear_all_rag_data(self):
        """Supprime physiquement tous les dossiers de base de données et réinitialise le service."""
        debug_log = []
        try:
            # 1. On coupe les connexions
            self.vector_db = None
            self.retriever = None
            self.chain = None
            gc.collect()
            time.sleep(1)

            # 2. On cherche et on supprime TOUS les dossiers chroma_db*
            dirs_to_delete = glob.glob("./chroma_db*")
            for folder in dirs_to_delete:
                shutil.rmtree(folder, ignore_errors=True)
                debug_log.append(f"Supprimé : {folder}")

            # 3. On réinitialise une base vide par défaut
            self.persist_dir = "./chroma_db"
            self.initialize_rag()
            
            return True, "Base de données nettoyée intégralement."
        except Exception as e:
            return False, f"Erreur lors du nettoyage : {str(e)}"
        
    def get_chunks_by_path(self, path: str):
        """Récupère les segments de texte associés à un chemin de fichier spécifique."""
        try:
            # 1. Normalisation
            normalized_path = os.path.normpath(path).replace("\\", "/")
            
            # 2. Tentative de recherche directe
            results = self.vector_db.get(
                where={"source": path}, 
                include=["documents"]
            )
            
            # 3. Si vide, tentative avec le chemin normalisé
            if not results.get("documents"):
                results = self.vector_db.get(
                    where={"source": normalized_path},
                    include=["documents"]
                )

            # 4. Si toujours vide, filtrage manuel par suffixe
            if not results.get("documents"):
                all_data = self.vector_db.get(include=["documents", "metadatas"])
                all_docs = all_data.get("documents", [])
                all_metas = all_data.get("metadatas", [])
                
                # On compare si la fin du chemin stocké correspond au fichier demandé
                filtered_docs = [
                    doc for doc, meta in zip(all_docs, all_metas) 
                    if meta.get("source", "").replace("\\", "/").endswith(normalized_path)
                ]
                return filtered_docs

            return results.get("documents", [])
            
        except Exception as e:
            print(f"Erreur Service get_chunks_by_path: {e}")
            return []
