#Personal Research Assistant

A sophisticated AI-powered research assistant for comprehensive information discovery, collection, verification, and synthesis.

##Features

- **Topic Discovery**: Refine and expand research topics to cover essential aspects
- **Information Collection**: Search and retrieve content from multiple sources (web, Wikipedia, academic databases)
- **Verification**: Cross-check information from different sources to ensure reliability
- **Synthesis**: Combine and analyze collected information to generate insights
- **Reporting**: Create comprehensive research reports with proper citations

##Technology Stack

- **Python**: Core programming language
- **Streamlit**: User interface for research topic input, scope definition, and result visualization
- **Google Cloud**: Document analysis with Document AI and research storage with Cloud Storage
- **LangChain**: Integration of various tools (Google search, Wikipedia, academic databases) for research pipelines
- **LangGraph**: State graph for managing the research process (topic discovery → data collection → verification → synthesis → reporting)
- **LlamaIndex**: Efficient indexing of research sources for creating a queryable knowledge base
- **OpenAI**: LLM for understanding research material, summarization, and coherent report generation

##Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/personal-research-assistant.git
cd personal-research-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
```

##Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser at `http://localhost:8501`

3. Enter your research topic, select the scope, and start the research process

4. View the results, explore the collected information, and download the generated report

##Configuration

You can customize the research assistant by modifying the settings in `config.py`:

- Adjust the OpenAI model parameters
- Configure search depth and breadth
- Set up preferred academic databases
- Customize report formatting

##Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##License

This project is licensed under the MIT License - see the LICENSE file for details.
