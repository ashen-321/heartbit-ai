import logging
from typing import List, Optional
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import httpx
from fastapi import HTTPException
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp_server = FastMCP(name="Pubmed Server", host="0.0.0.0", port=8001)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    """Structure for PubMed article data"""
    pmid: str
    title: str
    authors: List[str]
    abstract: str
    doi: Optional[str]
    publication_date: Optional[str]
    journal: Optional[str]
    url: str


class PubMedSearchRequest(BaseModel):
    query: str
    max_results: int = 10
    sort: str = "relevance"
    date_range: Optional[str] = None


class PubMedServer:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.http_client = httpx.AsyncClient(timeout=30.0)

    @mcp_server.tool()
    async def search_pubmed(self, request: PubMedSearchRequest) -> List[PubMedArticle]:
        """Search PubMed database"""
        try:
            # First, search for PMIDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": request.query,
                "retmax": request.max_results,
                "retmode": "json",
                "sort": request.sort
            }
            
            if request.date_range:
                search_params["datetype"] = "pdat"
                search_params["reldate"] = request.date_range
                
            search_response = await self.http_client.get(search_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []
                
            # Fetch detailed information for each PMID
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract"
            }
            
            fetch_response = await self.http_client.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()
            
            return self._parse_pubmed_xml(fetch_response.text)
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            raise HTTPException(status_code=500, detail=f"PubMed search failed: {str(e)}")

    def _parse_pubmed_xml(self, xml_data: str) -> List[PubMedArticle]:
        """Parse PubMed XML response"""
        articles = []
        try:
            root = ET.fromstring(xml_data)
            
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    # Extract PMID
                    pmid_elem = article_elem.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    # Extract title
                    title_elem = article_elem.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title available"
                    
                    # Extract authors
                    authors = []
                    author_list = article_elem.find(".//AuthorList")
                    if author_list is not None:
                        for author in author_list.findall(".//Author"):
                            last_name = author.find("LastName")
                            first_name = author.find("ForeName")
                            if last_name is not None:
                                name = last_name.text
                                if first_name is not None:
                                    name += f", {first_name.text}"
                                authors.append(name)
                    
                    # Extract abstract
                    abstract_elem = article_elem.find(".//Abstract/AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                    
                    # Extract DOI
                    doi = None
                    for article_id in article_elem.findall(".//ArticleId"):
                        if article_id.get("IdType") == "doi":
                            doi = article_id.text
                            break
                    
                    # Extract publication date
                    pub_date = None
                    pub_date_elem = article_elem.find(".//PubDate")
                    if pub_date_elem is not None:
                        year = pub_date_elem.find("Year")
                        month = pub_date_elem.find("Month")
                        if year is not None:
                            pub_date = year.text
                            if month is not None:
                                pub_date += f"-{month.text}"
                    
                    # Extract journal
                    journal_elem = article_elem.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else None
                    
                    article = PubMedArticle(
                        pmid=pmid,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        doi=doi,
                        publication_date=pub_date,
                        journal=journal,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                    )
                    articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
                    
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            
        return articles


if __name__ == "__main__":
    try:
        mcp_server.run(transport='streamable-http')
    except KeyboardInterrupt:
        pass
    finally:
        logger.info('Pubmed MCP server successfully shut down.')
