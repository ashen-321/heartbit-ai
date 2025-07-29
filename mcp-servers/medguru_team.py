from agno.agent import Agent
import os
import re
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters
from textwrap import dedent
from agno.tools.mcp import MultiMCPTools
from agno.models.openai import OpenAIChat
from langchain_openai import ChatOpenAI
# from agno.models.openai import OpenAIChat
import asyncio
from agno.knowledge.url import UrlKnowledge
# from agno.models.openai import OpenAIChat
from agno.models.aws.bedrock import AwsBedrock
from agno.storage.sqlite import SqliteStorage
from agno.team import Team
from agno.tools.tavily import TavilyTools
from agno.tools.knowledge import KnowledgeTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.thinking import ThinkingTools
# from agno.vectordb.lancedb import LanceDb, SearchType
from agno.vectordb.chroma import ChromaDb
import configparser
from agno.models.vllm import vLLM
from agno.models.openai import OpenAIChat  # could be OpenRouter as well
from openai import OpenAI
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.aws_bedrock import AwsBedrockEmbedder

ai_thinking_prompt = """
        You are a seasoned medical care provider specializing in diagnosis, prescribe medicine, analyzing medical and bio research papers from pubmed and medrxiv. In addition you can accurately lookup diagnosis code and procedure codes from ICD-10 and recommend clinical trails.

        Follow these steps for comprehensive diagnosis, pathological anaylsis, medication prescription and medical research includinh potential clinical trail recommendation:

        Your diagnosis style:
        - Thorough history taking: Gathering detailed information about the patient's symptoms, medical history, family history, and lifestyle factors.
        - Comprehensive physical examination: Conducting a systematic examination tailored to the patient's complaints to identify physical signs and abnormalities.
        - Appropriate diagnostic testing: Ordering relevant laboratory tests, imaging studies, or specialized procedures based on the differential diagnosis.
        - Integration and synthesis of information: Analyzing all collected data to develop a prioritized list of possible diagnoses (differential diagnosis).
        - Continuous reassessment: Evaluating the patient's response to treatment, updating the diagnosis if needed, and maintaining open communication with the patient throughout the process.

        Prescribe Accurate Medication:
        - Complete patient assessment: Gathering comprehensive information about the patient's condition, symptoms, medical history, current medications, allergies, and previous adverse drug reactions.
        - Establish accurate diagnosis: Confirming the correct diagnosis through appropriate examination and testing before determining medication needs.
        - Consider patient-specific factors: Accounting for age, weight, kidney and liver function, pregnancy status, genetics, and other individual factors that may affect drug metabolism and dosing.
        - Evidence-based selection: Choosing medications based on current clinical guidelines, scientific evidence for effectiveness, and consideration of risk-benefit profiles for the specific condition.
        - Monitor and adjust: Implementing appropriate follow-up to evaluate medication effectiveness, identify adverse effects, and make necessary adjustments to the treatment plan based on patient response.

        Pathological finding:
        - Proper specimen collection and handling: Ensuring appropriate collection, labeling, preservation, and transportation of tissue or fluid samples to maintain specimen integrity.
        - Comprehensive gross examination: Carefully observing and documenting the macroscopic features of the specimen, including size, color, consistency, and any visible abnormalities.
        - Systematic histological processing and examination: Preparing high-quality microscopic slides through proper fixation, processing, sectioning, and staining, followed by detailed microscopic evaluation.
        - Integration with clinical information: Correlating pathological findings with the patient's clinical history, symptoms, laboratory results, and imaging studies for context.
        - Application of specialized techniques when needed: Utilizing additional diagnostic methods such as immunohistochemistry, molecular testing, flow cytometry, or electron microscopy to confirm diagnoses or provide prognostic information.

        Recommend Clinical Trials:
        - Assess patient eligibility and suitability: Thoroughly evaluate the patient's condition, disease stage, previous treatments, comorbidities, and personal circumstances to determine appropriate trial matches.
        - Stay informed about available trials: Maintain current knowledge of ongoing clinical trials through research databases, professional networks, institutional review boards, and trial registries like ClinicalTrials.gov.
        - Clearly explain trial details and implications: Provide comprehensive information about the trial's purpose, treatment protocol, potential benefits, risks, side effects, required time commitment, and how it differs from standard care.
        - Facilitate informed consent: Ensure patients fully understand the voluntary nature of participation, their rights to withdraw, randomization procedures if applicable, and help them weigh potential benefits against risks.
        - Coordinate seamless integration with care team: Establish clear communication channels between the patient's regular healthcare providers and the clinical trial team to maintain continuity of care and monitor for adverse events.

        Match Accurate ICD-10 Disease and Treatment Codes:
        - Document comprehensive clinical information: Record detailed patient symptoms, examination findings, test results, and final diagnosis to support accurate code selection.
        - Identify the most specific diagnosis code: Locate the most precise ICD-10 code that matches the confirmed diagnosis, utilizing the full code structure including category, etiology, anatomic site, and severity/stage when applicable.
        - Include all relevant secondary diagnoses: Document and code for comorbidities, complications, and other conditions that affect patient care or treatment decisions.
        - Select appropriate procedure and treatment codes: Match treatments, procedures, and services provided to the correct CPT or procedure codes that precisely reflect the interventions performed.
        - Perform regular coding audits: Review coding practices periodically, stay updated with ICD-10 and procedure code changes, and utilize coding software or specialists when needed to ensure compliance and accuracy.
    """

# Tools
agno_docs = UrlKnowledge(
    urls=["https://www.paulgraham.com/read.html"],
    vector_db=ChromaDb(
        persistent_client=True,
        path="tmp/chroma_db",
        collection="agno_docs",
        reranker=None,
        embedder=OpenAIEmbedder(id="text-embedding-3-small")
    )
)

knowledge_tools = KnowledgeTools(
    knowledge=agno_docs,
    think=True,
    search=True,
    analyze=True,
    add_few_shot=True,
)

sequential_thinking_mcp_tools = MCPTools(
    command="npx -y @modelcontextprotocol/server-sequential-thinking"
)

medrxiv_streamable_url = "http://0.0.0.0:8083/mcp"  # "http://infs.cavatar.info:8083/mcp"

server_params_healthcare = StdioServerParameters(
    command="npx",
    args=["-y", "/home/alfred/codes/agent/MCP/healthcare-mcp-public"],
)
server_params_sthinking = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
)

# Models
model_id_c37 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
model_id_h35 = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'
model_id_c35 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
model_id_nova = 'us.amazon.nova-lite-v1:0'
model_id_ds = 'us.deepseek.r1-v1:0'


## AWS Credentials
def get_aws_credentials_from_file():
    try:
        # Path to the AWS credentials file
        aws_credentials_path = os.path.expanduser("~/.aws/credentials")

        # Create a ConfigParser object
        config = configparser.ConfigParser()

        # Read the credentials file
        config.read(aws_credentials_path)

        # Extract the access key ID and secret access key from the [default] profile
        if 'default' in config:
            aws_access_key_id = config['default'].get('aws_access_key_id')
            aws_secret_access_key = config['default'].get('aws_secret_access_key')
            return aws_access_key_id, aws_secret_access_key
        else:
            print("No [default] profile found in credentials file")
            return None, None

    except Exception as e:
        print(f"Error reading AWS credentials: {e}")
        return None, None


# os.environ["OPENAI_API_KEY"] = openai_api_key = os.getenv("bedrock_api_token")
# os.environ["OPENAI_BASE_URL"] = openai_base_url = os.getenv("bedrock_api_url")
os.environ["TAVILY_API_KEY"] = tavily_api_key = os.getenv("tavily_api_token")
os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"] = get_aws_credentials_from_file()

# Standard LLM model (e.g., GPT-4 via OpenRouter/OpenAI)
# model = OpenAIChat(id=model_id_nova)  # assumes API key set via env
# model = Claude(id=model_id_c37, aws_region="us-west-2")  # assumes API key set via env
model_nova = AwsBedrock(id=model_id_nova)
model_c37 = AwsBedrock(id=model_id_c37)
model_h35 = AwsBedrock(id=model_id_h35)
model_ds = AwsBedrock(id=model_id_ds)
model_qwen = vLLM(base_url="http://agent.cavatar.info:8081/v1", api_key="EMPTY", id="Qwen/Qwen3-30B-A3B",
                  temperature=0.2, top_p=0.90, presence_penalty=1.45)
model_gemma = vLLM(base_url="http://infs.cavatar.info:8081/v1", api_key="EMPTY",
                   id="alfredcs/gemma-3-27b-grpo-med-merged", temperature=0.2, top_p=0.90, presence_penalty=1.45)

model_openai_br = ChatOpenAI(
    model=model_id_c37,
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=5,
)

model_openai = OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("openai_api_token"),
                          base_url="https://api.openai.com/v1/")

agent_storage_file: str = "tmp/agents.db"
image_agent_storage_file: str = "tmp/image_agent.db"

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Agents
cot_agent = Agent(
    name="Chain-of-Thought Agent",
    role="Answer basic questions",
    agent_id="cot-agent",
    model=model_ds,
    storage=SqliteStorage(
        table_name="cot_agent", db_file=agent_storage_file, auto_upgrade_schema=True
    ),
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    markdown=True,
    reasoning=True,
)

reasoning_model_agent = Agent(
    name="Reasoning Model Agent",
    role="Reasoning about Math",
    agent_id="reasoning-model-agent",
    model=model_ds,
    reasoning_model=model_gemma,
    instructions=["You are a reasoning agent that can reason about math."],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    storage=SqliteStorage(
        table_name="reasoning_model_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)

reasoning_tool_agent = Agent(
    name="Reasoning Tool Agent",
    role="Answer basic questions",
    agent_id="reasoning-tool-agent",
    model=model_ds,
    storage=SqliteStorage(
        table_name="reasoning_tool_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    markdown=True,
    tools=[ReasoningTools()],
)

web_agent = Agent(
    name="Web Search Agent",
    role="Handle web search requests",
    model=model_ds,
    agent_id="web_agent",
    tools=[TavilyTools()],
    instructions="Always include sources",
    add_datetime_to_instructions=True,
    storage=SqliteStorage(
        table_name="web_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
    stream=True,
    stream_intermediate_steps=True,
)

thinking_tool_agent = Agent(
    name="Thinking Tool Agent",
    agent_id="thinking_tool_agent",
    model=model_qwen,
    # reasoning_model=None,
    # reasoning=False,
    tools=[ThinkingTools(add_instructions=True)],
    instructions=dedent(ai_thinking_prompt),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
    stream_intermediate_steps=True,
    storage=SqliteStorage(
        table_name="thinking_tool_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)

knowledge_agent = Agent(
    agent_id="knowledge_agent",
    name="Knowledge Agent",
    model=model_ds,
    tools=[knowledge_tools],
    show_tool_calls=True,
    markdown=True,
    storage=SqliteStorage(
        table_name="knowledge_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)

# async with MCPTools(url=medrxiv_streamable_url, transport="streamable-http", timeout_seconds=60) as stream_medrxiv_mcp_tools:
medrxiv_agent = Agent(
    agent_id="medrxiv_agent",
    name="Medrxiv Agent",
    model=model_qwen,
    tools=[MCPTools(url=medrxiv_streamable_url, transport="streamable-http", timeout_seconds=60)],
    show_tool_calls=True,
    markdown=True,
    # reasoning_model=None,
    # reasoning=False,
    instructions=dedent("""\
        You are a medical publication specialist who can search papers and answer patients' questions about diagnosis, treatment, medication, clinical trails and medical publications.
        - Keep the url links as is and return to your answer. Don't modify or generate any url address.
        - Be concise and focus on relevant information\
    """),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=7,
    storage=SqliteStorage(
        table_name="knowledge_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)

healthcare_agent = Agent(
    agent_id="healthcare_agent",
    name="Healthcare Agent",
    model=model_qwen,
    tools=[MCPTools(server_params=server_params_healthcare, transport="stdio", timeout_seconds=60)],
    show_tool_calls=True,
    markdown=True,
    # reasoning_model=None,
    # reasoning=False,
    instructions=dedent("""\
        You are a seasoned care provider who can search papers and diagnose and treamt disease and answer patients' questions about medication, clinical trails and medical publications.
        You can map disease and treatment codes from doctor's note using ICD-10-CM and ICD-10-PCS standard.
        - Keep the url links as is and return to your answer. Don't modify or generate any url address.
        - Be concise and focus on relevant information\
    """),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=7,
    storage=SqliteStorage(
        table_name="healthcare_agent",
        db_file=agent_storage_file,
        auto_upgrade_schema=True,
    ),
)


# Call agents or team
async def call_med_agents(prompt: str, stream: bool = True, stream_intermediate_steps: bool = True):
    # Create a client session to connect to the MCP server
    # async with MultiMCPTools([server_params_github, server_params_sthinking]) as mcp_tools: #Doesn't work
    async with MultiMCPTools(
            urls=[medrxiv_streamable_url],
            urls_transports=["streamable-http"],  # Same length as urls
            server_params_list=[server_params_healthcare, server_params_sthinking],
            include_tools=None,
            # [ReasoningTools(add_instructions=True), ThinkingTools(add_instructions=True)], #was None
            exclude_tools=None,
            timeout_seconds=120,
            # MCPTools(url=github_mcp_server_url, transport="streamable-http", timeout_seconds=120) as stream_github_mcp_tools,
            # MCPTools(server_params=server_params_github ) as local_github_mcp_tools,
            # MCPTools(server_params=server_params_sthinking ) as local_sthinking_mcp_tools,
    ) as multi_mcp_tools:
        med_agents = Agent(
            model=model_qwen,
            # model_openai works fine. model_qwen works, so does model_gemma but not model_nova nor model_openai_br
            tools=[multi_mcp_tools],
            instructions=dedent("""
                    You are a medical doctor assistant agent who can answer patients' questions about diagnosis, treatment, medication, clinical trails and medical publications.
                    - Aggregate answers from all agents and construct a comprehensive and accurate response
                    - Accuracy is the most important quality of your response
                    - Keep the url links as is and return to your answer. Don't modify or generate any url address.
                    - Be concise and focus on relevant information
                """),
            reasoning_model=model_qwen,
            reasoning=True,
            reasoning_max_steps=7,
            markdown=True,
            show_tool_calls=True,
            debug_mode=False,
            # stream=True,
            # stream_intermediate_steps=True,
            add_datetime_to_instructions=True,
            add_history_to_messages=True,
            num_history_responses=7,
        )
        # Run the agent
        responses = await med_agents.arun(prompt, stream=stream, stream_intermediate_steps=stream_intermediate_steps)
        return re.sub(r'timer=<agno\.utils\.timer\.Timer object at 0x[0-9a-fA-F]+>', '', str(responses))


async def call_med_team(prompt: str, stream: bool = True, stream_intermediate_steps: bool = True):
    reasoning_med_team = Team(
        name="Reasoning Medical Team",
        mode="coordinate",
        model=model_qwen,
        # reasoning_model=None,
        members=[
            medrxiv_agent,
            healthcare_agent,
            thinking_tool_agent,
            cot_agent,
            reasoning_tool_agent,
            reasoning_model_agent,
        ],
        # reasoning=False,  # Does not work with model_qwen??
        # reasoning_max_steps=7,
        tools=[ReasoningTools(add_instructions=True)],
        # uncomment it to use knowledge tools
        # tools=[knowledge_tools],
        team_id="reasoning-medical-team",
        debug_mode=False,
        instructions=dedent("""
        You are a team of medical doctor assistants who can answer patients' questions about diagnosis, treatment, medication, clinical trails and medical publications.
                - Gather all agents' findings and form a comprehensive and accurate answer
                - Cite sources for any facts and maintain clarity in the final answer
                - Keep the url links as is and return to your answer. Don't modify or generate any url address.
                - Be concise and focus on relevant information
                - Use tables to display data, if any
                - Always check the conversation history (memory) for context or follow-up references
                - If the user asks something that was asked before, utilize remembered information instead of starting fresh
                - Continue delegating and researching until the query is fully answered
        """),
        markdown=True,
        add_datetime_to_instructions=True,
        success_criteria="All agents have successfully completed all the designated tasks. The team has completed result aggregation from all agents and generated a comprehensive response.",
        storage=SqliteStorage(
            table_name="reasoning_medical_team",
            db_file=agent_storage_file,
            auto_upgrade_schema=True,
        ),
        # Newly added
        enable_agentic_context=True,  # The coordinator retains its own context between turns
        share_member_interactions=True,  # All agents see each other's outputs as context
        show_members_responses=True,  # Do not show raw individual agents' answers directly to the user
        enable_team_history=True,  # Maintain a shared history (memory) between coordinator and members
        num_of_interactions_from_history=5  # Limit how much history is shared (to last 5 interactions)
    )

    responses = await reasoning_med_team.arun(prompt, stream=stream, stream_intermediate_steps=stream_intermediate_steps)
    # return await reasoning_finance_team.aprint_response(prompt, stream=stream, stream_intermediate_steps=stream_intermediate_steps)
    return re.sub(r'timer=<agno\.utils\.timer\.Timer object at 0x[0-9a-fA-F]+>', '', str(responses))


if __name__ == "__main__":
    prompt = "List the latest publications on the prevention of diabetes mellitus. Prescribe available medication and suggest clinical trials for treatment."
    messages = [
        {"role": "system",
         "content": "You are a helpful medical assistant. Please answer the user question accurately and truthfully. Also please make sure to think carefully before answering"},
        {"role": "user", "content": prompt},
    ],
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(call_mcp(prompt))

    results = asyncio.run(call_med_team(prompt, False, False))
    print(f"{results}")
