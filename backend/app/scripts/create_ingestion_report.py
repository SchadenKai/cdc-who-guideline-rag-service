import uuid
from app.data.test_web_sources import urls
from app.routes.dependencies.rag import get_indexing_service_manual

md_report_template = """
# Ingestion Report
Contains 10 URLs from both WHO and CDC  
"""

indexing_agent = get_indexing_service_manual()
for url in urls:
    res_state = indexing_agent.ingest_document(website_url=url, request_id=str(uuid.uuid4()))