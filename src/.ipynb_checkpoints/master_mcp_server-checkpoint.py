from mcp_master import MasterMCPServer, SubServer
from mcp_master import global_config as gconfig
from os import getenv, path

gconfig.selector_model_id = 'alfredcs/gemma-3-27b-grpo-med-merged'
gconfig.judge_model_id = 'alfredcs/torchrun-medgemma-27b-grpo-merged'
gconfig.judge_model_service_url = 'http://mcp1.cavatar.info:8081'
gconfig.OPENAI_API_KEY = getenv('bedrock_api_token')
gconfig.OPENAI_BASE_URL = 'http://infs.cavatar.info:8081' #getenv('bedrock_api_url')
gconfig.autostart_abspath = path.normpath(path.join(path.dirname(__file__), '../mcp-servers'))

master_server = MasterMCPServer(
    port=8089,
    sub_servers=[
        SubServer(url="http://localhost:8001/mcp", identifier='pubmed_server'),
        SubServer(url="http://localhost:8002/mcp", identifier='medrxiv_server'),
        # SubServer(url="http://localhost:8003/mcp", identifier='medguru_server'),
    ]
)
master_server.startup()
