# Personal Research Assistant

An AI-powered research assistant designed for efficient information discovery, collection, verification, and synthesis.

## Features

- **Topic Discovery**: Expand and refine research topics to ensure comprehensive coverage.
- **Information Collection**: Retrieve content from multiple sources, including the web, Wikipedia, and academic databases.
- **Verification**: Cross-check information from different sources for reliability and accuracy.
- **Synthesis**: Analyze and integrate collected data to generate meaningful insights.
- **Reporting**: Generate well-structured research reports with proper citations.

## Technology Stack

- **Python**: Core programming language.
- **Streamlit**: Provides a user-friendly interface for topic input, scope selection, and result visualization.
- **Google Cloud**: Utilizes Document AI for document analysis and Cloud Storage for research data.
- **LangChain**: Manages research pipelines, integrating tools like Google Search, Wikipedia, and academic databases.
- **LangGraph**: Implements a state graph to manage research flow (topic discovery → data collection → verification → synthesis → reporting).
- **LlamaIndex**: Enables efficient indexing of research sources for quick querying.
- **OpenAI**: Powers language models for research understanding, summarization, and coherent report generation.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/personal-research-assistant.git
cd personal-research-assistant
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file and add the following configurations:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
```

## Usage

### 1. Run the Application
```bash
streamlit run app.py
```

### 2. Access the Interface
Open your web browser and navigate to:  
`http://localhost:8501`

### 3. Start Researching
- Enter your research topic.
- Define the scope and parameters.
- Initiate the research process.

### 4. Explore and Download Results
- View collected information.
- Analyze findings.
- Download the generated research report.

## Configuration

Customize the research assistant by modifying the settings in `config.py`:

- Adjust OpenAI model parameters.
- Configure search depth and scope.
- Define preferred academic databases.
- Customize report formatting options.

## Contributing

We welcome contributions! Feel free to submit a pull request to enhance the project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
