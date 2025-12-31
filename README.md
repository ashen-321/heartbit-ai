# Heartbit AI

A modern healthcare AI application that provides intelligent medical assistance through a chatbot interface powered by large language models and specialized medical knowledge servers.

## Description

Heartbit AI is an advanced medical assistant application that leverages the Model Context Protocol (MCP) to provide accurate medical information and first aid guidance. The system integrates multiple specialized servers to access:

- **PubMed Database**: Search and retrieve peer-reviewed medical research articles
- **medRxiv Preprints**: Access the latest medical research preprints
- **ICD-10 Codes**: Query the International Classification of Diseases database

The application offers two interaction modes:
1. **Streamlit Web Interface** (`home.py`): A user-friendly web application with image upload, voice input, and rich text support
2. **Terminal Chatbot** (`terminal_chatbot.py`): A command-line interface for quick medical queries

## Architecture

The system is built on a master MCP server architecture:

- **Master MCP Server**: Orchestrates multiple specialized MCP servers and routes requests
- **MCP Sub-Servers**:
  - PubMed Server (port 8001): Searches medical literature
  - medRxiv Server (port 8002): Retrieves preprint medical research
  - ICD-10 Server (port 8003): Queries disease classification codes
- **AI Models**: Supports multiple LLM backends including Gemma, Claude, and custom fine-tuned medical models

## Features

- Medical question answering with context-aware responses
- Image upload for visual medical queries (rash analysis, injury assessment, etc.)
- Voice input for hands-free querying
- Real-time medical literature search
- ICD-10 diagnostic code lookup
- Streaming responses for lower latency
- Session history management
- Token usage tracking and latency monitoring

## Requirements

### System Requirements
- Python 3.11+
- Internet connection for API access

### Python Dependencies

```bash
mcp-master
streamlit
fastapi
bs4
requests
```

Run `pip install -r requirements.txt` to install all dependencies.

### Environment Variables

Create a `.aoss_config.txt` file (optional, for OpenSearch integration):
```
AOSS_host_name: your-opensearch-host
AOSS_index_name: your-index-name
```

Set the following environment variables to invoke backend models through AWS Bedrock:
```bash
export bedrock_api_token="your-api-token"
export bedrock_api_url="your-api-url"
```

## Getting Started

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ashen-321/heartbit-ai.git
cd heartbit-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API endpoints** (Optional)

Edit `src/master_mcp_server.py` to configure your LLMs and their corresponding endpoints:
```python
gconfig.OPENAI_API_KEY = "your-api-key"
gconfig.OPENAI_BASE_URL = "your-api-url"
```

### Running the Application

#### Step 1: Start the Master MCP Server

```bash
cd src
python master_mcp_server.py
```

This will start:
- Master server on port 8089
- PubMed server on port 8001
- medRxiv server on port 8002
- ICD-10 server on port 8003

#### Step 2A: Launch the Web Interface

In a new terminal:
```bash
cd src
streamlit run home.py
```

Access the application at `http://localhost:8501`

**OR**

#### Step 2B: Use the Terminal Chatbot

In a new terminal:
```bash
cd src
python terminal_chatbot.py
```

## Configuration

### Model Selection

You can configure different models in `src/master_mcp_server.py`:

```python
# Available model options:
model_id_c37 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
model_id_c35 = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
model_id_nova = "us.amazon.nova-lite-v1:0"
model_id_llama = "meta.llama3-3-70b-instruct-v1:0"

# Set your preferred model:
gconfig.selector_model_id = model_id_c35
```

### Web Interface Settings

In the web interface sidebar, you can adjust:
- Maximum output tokens (0-4096)
- Upload images for visual analysis
- Enable voice input
- Clear chat history


## Usage Examples

### Web Interface

1. **Simple medical query**:
   - Type: "What are the symptoms of diabetes?"
   - The AI will provide information and cite relevant research if needed

2. **Image analysis**:
   - Upload an image of a rash or injury in the sidebar
   - Ask: "What could this rash indicate?"
   - The AI analyzes the image and provides assessment

3. **Voice input**:
   - Click the microphone icon in the sidebar
   - Speak your question
   - The system transcribes and processes your query

### Terminal Interface

```bash
Query: What is the ICD-10 code for type 2 diabetes?
# System searches ICD-10 database and returns relevant codes

Query: Search for recent research on COVID-19 vaccines
# System queries PubMed and medRxiv for relevant articles

Query: wipe
# Clears conversation history

Query: quit
# Exits the application
```

## Sample Queries

**General Medical Questions:**
- "What are the early signs of a heart attack?"
- "How do I treat a minor burn at home?"
- "What's the difference between Type 1 and Type 2 diabetes?"

**Research Queries:**
- "Find recent studies on immunotherapy for cancer"
- "What's the latest research on Alzheimer's prevention?"
- "Search for articles about mRNA vaccine effectiveness"

**Diagnostic Code Lookup:**
- "What's the ICD-10 code for hypertension?"
- "Find codes related to chronic kidney disease"
- "ICD-10 codes for anxiety disorders"

**Image-Based Queries** (Web interface only):
- Upload image + "Is this rash concerning?"
- Upload image + "What type of wound is this?"
- Upload image + "Does this bruising pattern indicate anything serious?"

## Project Structure

```
heartbit-ai/
├── src/
│   ├── home.py                    # Streamlit web interface
│   ├── terminal_chatbot.py        # Command-line interface
│   ├── master_mcp_server.py       # Master server configuration
│   └── util.py                    # Utility functions
├── mcp-servers/
│   ├── pubmed_server.py           # PubMed search server
│   ├── medrxiv_server.py          # medRxiv search server
│   ├── medrxiv_web_search.py      # medRxiv web scraping utilities
│   └── icd10_server.py            # ICD-10 code lookup server
├── input-files/                   # User-uploaded files directory
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Roadmap

### Phase 1: Core Enhancements
- [ ] Add support for more medical databases (OMIM, DrugBank, ClinicalTrials.gov)
- [ ] Implement user authentication and session persistence
- [ ] Improve error handling and retry logic

### Phase 2: Advanced Features
- [ ] Add support for DICOM medical imaging analysis
- [ ] Develop symptom checker with decision tree guidance
- [ ] Add export functionality (PDF reports, email summaries)

### Phase 3: Clinical Integration
- [ ] Implement HIPAA-compliant data handling
- [ ] Add audit logging for clinical use

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Heartbit AI is designed for educational and informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding any medical conditions. Never disregard professional medical advice or delay seeking it because of information from this application.

## Acknowledgments

- PubMed/NIH for providing access to medical literature
- medRxiv for preprint research access
- Model Context Protocol (MCP) framework
- Anthropic Claude and Google Gemma teams for LLM technology

## Support

For issues, questions, or suggestions, please:
- Open an issue on GitHub
- Contact the repository maintainers

## Citation

If you use Heartbit AI in your research, please cite:

```bibtex
@software{heartbit_ai_2025,
  author = {ashen-321},
  title = {Heartbit AI: A Modern Healthcare AI Application},
  year = {2025},
  url = {https://github.com/ashen-321/heartbit-ai}
}
```
