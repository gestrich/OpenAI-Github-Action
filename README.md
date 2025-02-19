# OpenAI GitHub Action

This GitHub Action analyzes code changes in pull requests using OpenAI's GPT models to provide code improvement suggestions.

## Local Development Setup

### Prerequisites
- Python 3.x
- Git

### Virtual Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   
   Your prompt should change to show `(.venv)` indicating the virtual environment is active.

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. When you're done, you can deactivate the virtual environment:
```bash
deactivate
```

### Configuration

1. Set your OpenAI API key either:
   - As an environment variable:
     ```bash
     export OPENAI_API_KEY='your-api-key'
     ```
   - Or in a configuration file at `~/.aiScriptConfiguration.json`:
     ```json
     {
         "openai": {
             "api_key": "your-api-key-here",
             "model": "gpt-4"
         }
     }
     ```

### Running the Script

Remember to activate the virtual environment before running the script:
```bash
source .venv/bin/activate  # On macOS/Linux
```

Then you can run the script:

To analyze the current changes:
```bash
python diffAnalyzer.py
```

To analyze a specific commit:
```bash
python diffAnalyzer.py <commit-hash>
```

To enable PR annotations (when running in GitHub Actions):
```bash
python diffAnalyzer.py --annotate-pr
```

## GitHub Action Usage

1. Add your OpenAI API key as a repository secret named `OPENAI_API_KEY`

2. The action will automatically run on pull requests to the main branch

3. View the analysis results:
   - In the GitHub Actions run summary
   - As PR comments (if annotations are enabled)

## Development Notes

- The script uses GPT-4 by default
- Analysis includes:
  - Code quality issues
  - Potential bugs
  - Performance improvements
  - Best practices
- Language-specific analysis for:
  - Objective-C
  - Swift
  - C++

## Project Structure

- `diffAnalyzer.py`: Main analysis script
- `git_diff.py`: Git diff parsing utilities
- `.github/workflows/run-script.yml`: GitHub Action workflow definition
- `requirements.txt`: Python dependencies

## Dependencies

- openai>=1.0.0
- langchain
- langchain-openai
- pydantic>=2.0.0

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: If you see errors about missing modules, make sure:
   - You've activated the virtual environment
   - You've installed the dependencies
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **External Environment Error**: If you see "externally-managed-environment" error:
   - Make sure you're using the virtual environment
   - The virtual environment must be activated before running pip install

3. **API Key Issues**: If you get authentication errors:
   - Check that your OpenAI API key is properly set
   - Verify the configuration file format
   - Try using the environment variable instead


