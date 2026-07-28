from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db_models import DocumentStatus
from app.rag.types import RetrievedChunk, TextChunk

def search_chunks(db:Session, query_vector:list[float], query_text:str, limit:int)->list[RetrievedChunk]:
    sql=text("""
    WITH ranked AS (
      SELECT c.id,c.page_number,c.content_type,c.text,d.filename,
             1-(c.embedding <=> CAST(:embedding AS vector)) AS vector_score,
             ts_rank_cd(to_tsvector('simple',c.text),plainto_tsquery('simple',:query)) AS keyword_score
      FROM document_chunks c JOIN documents d ON d.id=c.document_id
      WHERE d.status='ready'
    )
    SELECT *, LEAST(1.0,GREATEST(0.0,vector_score*0.75+keyword_score*0.25)) AS score
    FROM ranked ORDER BY score DESC LIMIT :limit
    """)
    rows=db.execute(sql,{"embedding":str(query_vector),"query":query_text,"limit":limit}).mappings()
    return [RetrievedChunk(chunk=TextChunk(chunk_id=str(r["id"]),filename=r["filename"],page_number=r["page_number"],text=r["text"],content_type=r["content_type"]),score=float(r["score"]),method="pgvector+fts",vector_score=float(r["vector_score"]),keyword_score=float(r["keyword_score"] or 0)) for r in rows]
