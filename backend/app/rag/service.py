import logging
from collections import Counter
from datetime import UTC, datetime
from sqlalchemy import delete
from sqlalchemy.orm import Session
from app.config import get_settings
from app.db_models import Document, DocumentChunk, DocumentStatus
from app.models import AnswerResponse, SourceResult
from app.rag.chunking import chunk_pages
from app.rag.embeddings import embedding_service
from app.rag.guardrails import validate_grounded_answer
from app.rag.llm import llm_service
from app.rag.pdf import extract_pdf_pages
from app.rag.postgres_store import search_chunks
from app.rag.prompts import NO_ANSWER, build_user_prompt
from app.rag.query import query_planner
logger=logging.getLogger(__name__)
class RagService:
    def process_document(self,db:Session,document:Document)->Document:
        s=get_settings(); document.status=DocumentStatus.processing; document.error=None; db.commit()
        try:
            result=extract_pdf_pages(document.content,document.filename)
            chunks=chunk_pages(result.blocks,chunk_size=s.chunk_size_chars,overlap=s.chunk_overlap_chars)
            if not chunks: raise ValueError("No chunks were created")
            vectors=embedding_service.encode([c.text for c in chunks])
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id==document.id))
            for i,(chunk,vector) in enumerate(zip(chunks,vectors,strict=True)):
                db.add(DocumentChunk(document_id=document.id,chunk_index=i,page_number=chunk.page_number,content_type=chunk.content_type,text=chunk.text,embedding=vector.tolist()))
            document.page_count=result.total_pages; document.chunk_count=len(chunks); document.warnings=result.warnings; document.status=DocumentStatus.ready; document.processed_at=datetime.now(UTC); db.commit(); db.refresh(document); return document
        except Exception as exc:
            db.rollback(); document=db.get(Document,document.id); document.status=DocumentStatus.failed; document.error=str(exc); db.commit(); raise
    def ask(self,db:Session,question:str,top_k:int|None=None,rewrite_question:bool|None=None)->AnswerResponse:
        s=get_settings(); plan=query_planner.plan(question,enabled=rewrite_question); vectors=embedding_service.encode(plan.search_queries); limit=top_k or s.top_k
        merged={}
        for q,v in zip(plan.search_queries,vectors,strict=True):
            for item in search_chunks(db,v.tolist(),q,max(limit*3,12)):
                old=merged.get(item.chunk.chunk_id)
                if old is None or item.score>old.score: merged[item.chunk.chunk_id]=item
        relevant=sorted(merged.values(),key=lambda x:x.score,reverse=True)[:limit]
        if not relevant:return AnswerResponse(answer=NO_ANSWER,sources=[],grounded=False,grounding_status="insufficient_evidence",interpreted_question=plan.rewritten_question,search_queries=plan.search_queries)
        prompt,context=build_user_prompt(plan.original_question,plan.rewritten_question,relevant,s.max_context_chars)
        raw=llm_service.answer(prompt); answer,grounded=validate_grounded_answer(raw,len(context))
        sources=[SourceResult(id=f"S{i}",filename=x.result.chunk.filename,page=x.result.chunk.page_number,score=round(x.result.score,4),excerpt=x.excerpt,content_type=x.result.chunk.content_type,retrieval_method=x.result.method) for i,x in enumerate(context,1)]
        return AnswerResponse(answer=answer,sources=sources,grounded=grounded,grounding_status="verified" if grounded else "citation_validation_failed",interpreted_question=plan.rewritten_question,search_queries=plan.search_queries)
rag_service=RagService()
