from mcp_master import MasterMCPServer, SubServer
from mcp_master import global_config as gconfig
from os import getenv, path

model_id_c37 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
model_id_c35 = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
model_id_nova = "us.amazon.nova-lite-v1:0"
model_id_llama = "meta.llama3-3-70b-instruct-v1:0"

gconfig.selector_model_id = model_id_c35 #'alfredcs/gemma-3-27b-grpo-med-merged' #'alfredcs/torchrun-medgemma-27b-grpo-merged'
gconfig.judge_model_id = 'alfredcs/torchrun-medgemma-27b-grpo-merged'
gconfig.judge_model_service_url = 'http://mcp1.cavatar.info:8081/v1'
gconfig.OPENAI_API_KEY = getenv('bedrock_api_token')
gconfig.OPENAI_BASE_URL = getenv('bedrock_api_url') #'http://infs.cavatar.info:8081'
gconfig.autostart_abspath = path.normpath(path.join(path.dirname(__file__), '../mcp-servers'))

master_server = MasterMCPServer(
    port=8089,
    sub_servers=[
        SubServer(url="http://localhost:8001/mcp", identifier='pubmed_server'),
        SubServer(url="http://localhost:8002/mcp", identifier='medrxiv_server'),
        SubServer(url="http://localhost:8003/mcp", identifier='icd10_server'),
    ]
)
master_server.startup()
