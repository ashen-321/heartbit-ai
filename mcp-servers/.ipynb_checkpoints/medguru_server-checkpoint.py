import logging
from dataclasses import dataclass
from datetime import datetime

from fastmcp import FastMCP
from fastapi import FastAPI
from medguru_team import call_med_team, call_med_agents
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Track rate limit information"""
    remaining: int = 5000
    reset_time: datetime = datetime.now()
    limit: int = 5000


mcp = FastMCP("Team reasoning with multiple agents")
# Create the ASGI app
mcp_app = mcp.http_app(path='/mcp')

# Create a FastAPI app and mount the MCP server
app = FastAPI(lifespan=mcp_app.lifespan)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")
    

@mcp.tool()
async def team_reasoning(prompt: str, repo: str) -> str:
    """
    Call reasoning medical team to conduct medical diagnosis through AI-driven analysis of patient data (symptoms,
    lab results, imaging), prescribing medications with drug interaction checks and dosage guidance,
    searching real-time medical literature from PubMed and MedRxiv, retrieving ICD-10/CPT codes for billing,
    and identifying patient-specific clinical trial opportunities. Designed for accuracy and compliance,
    it streamlines workflows to enhance diagnostic confidence, reduce errors, and support personalized,
    evidence-based patient care.
    """
    try:
        logger.info(f"Seeking advisory on topic: {prompt}")
        results = await call_med_team(prompt, False, False)
        return results
    except Exception as e:
        error_msg = f"Error performing team reasoning for {prompt}: {e}"
        logger.error(error_msg)


@mcp.tool(description="Call reasoning medical agent to perform medical diagnosis through AI-driven analysis of patient data (symptoms, lab results, imaging), prescribing medications with drug interaction checks and dosage guidance, searching real-time medical literature from PubMed and MedRxiv, retrieving ICD-10/CPT codes for billing, and identifying patient-specific clinical trial opportunities. Designed for accuracy and compliance, it streamlines workflows to enhance diagnostic confidence, reduce errors, and support personalized, evidence-based patient care.")
async def agent_reasoning(prompt: str, repo: str) -> str:
    try:
        logger.info(f"Seeking advisory on topic: {prompt}")
        results = await call_med_agents(prompt, False, False)
        #return json.dumps(results, indent=2)
        return results
    except Exception as e:
        error_msg = f"Error performing team reasoning for {prompt}: {e}"
        logger.error(error_msg)
        

def main():
    """Run the MCP server for reasoning AI advisor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GitHub MCP Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8003, help='Port to bind to')
    parser.add_argument('--transport', default='streamable-http', 
                       choices=['streamable-http', 'http'], 
                       help='Transport protocol to use')
    
    args = parser.parse_args()
    
    logger.info(f"Starting GitHub MCP Server on {args.host}:{args.port}")
    logger.info(f"Transport protocol: {args.transport}")
    logger.info("Available tools:")
    logger.info("  - call team: Get team to reason collaboratively with multiple agents. Each agent has access to one or more tools")

    
    # Configure server for streamable HTTP transport
    if args.transport == 'streamable-http':
        logger.info("Configuring for streamable-http transport with SSE support")
        # Run the FastMCP server with streamable-http transport
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            # Additional configuration for streaming
            # stream_mode=True,
            # enable_sse=True,  # Server-Sent Events support
            # cors_enabled=True,  # Enable CORS for web clients
            # max_connections=100,
            # keepalive_timeout=300
        )
    else:
        # Fallback to regular HTTP transport
        mcp.run(transport="http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
