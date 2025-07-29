from mcp_master import MasterMCPServer, SubServer
from mcp_master import global_config as gconfig
from os import getenv, path

gconfig.selector_model_id = ''  # Set this to your tool selector model ID
gconfig.judge_model_id = ''  # Set this to your judge model ID
gconfig.judge_model_service_url = ''  # Set this to where your judge LLM is hosted
gconfig.OPENAI_API_KEY = getenv('OPENAI_API_KEY')
gconfig.OPENAI_BASE_URL = getenv('OPENAI_BASE_URL')
gconfig.autostart_abspath = path.normpath(path.join(path.dirname(__file__), '../mcp-servers'))

master_server = MasterMCPServer(
    port=3000,
    sub_servers=[
        SubServer(url="http://localhost:8001/mcp", identifier='pubmed_server'),
        SubServer(url="http://localhost:8002/mcp", identifier='medrxiv_server'),
    ]
)
master_server.startup()
