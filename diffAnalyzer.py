from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import sys
import json
import os
import argparse
from git_diff import GitDiff, get_git_diff

class CodeImprovement(BaseModel):
    """Model for code improvement suggestions."""
    file_path: str = Field(description="The file path where the issue occurs")
    line_number: int = Field(description="The line number where the issue occurs")
    description: str = Field(description="Clear description of the issue found")
    improvement: str = Field(description="Specific suggestion on how to improve the code")
    context: str = Field(default="", description="The surrounding code context")

def general_code_analysis_tips() -> str:
    """Return general tips and context for code analysis."""
    return """
Important Context for Code Analysis:
- This is a mixed codebase containing Objective-C, Swift, and C++
- For Objective-C code:
  * Sending messages to nil is safe and returns nil/zero
  * "Null pointer exceptions" are not relevant in Objective-C
  * Use of nil is part of normal control flow
- For Swift code:
  * Optionals are used for nil safety
  * Force unwrapping (!) should be used with caution
- For C++ code:
  * Null pointer dereference is a serious issue
  * RAII and smart pointers are preferred over raw pointers

Please consider these language-specific characteristics when analyzing the code.
"""

class DiffAnalyzer:
    """Main class for analyzing git diffs using OpenAI."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
            openai_api_key=api_key
        )
        self.setup_chains()
    
    def setup_chains(self):
        """Setup the analysis chain."""
        self.structured_llm = self.llm.with_structured_output(
            CodeImprovement,
            method="function_calling"
        )
        
        self.analysis_prompt = f"""
{general_code_analysis_tips()}

Below is a git diff with context, where each line is marked as:
- Lines starting with '+': Added or modified code
- Lines starting with '-': Removed code
- Other lines: Unchanged context

IMPORTANT: Only analyze the added/modified lines (starting with '+').
Use the context only to understand the changes better.

Focus on these aspects of the changed code:
1. Code quality issues
2. Potential bugs
3. Performance improvements
4. Best practices

For each suggestion:
1. Include the file path from the diff
2. Reference specific line numbers from the diff
3. Provide clear improvement suggestions
4. Consider the language-specific context

Code diff to analyze:
{{diff}}
"""

    def get_git_diff(self, commit_id: Optional[str] = None) -> Optional[GitDiff]:
        """Get the git diff for analysis."""
        return get_git_diff(commit_id)

    def analyze_diff(self, diff: GitDiff) -> List[CodeImprovement]:
        """Analyze the entire diff for improvements."""
        try:
            all_improvements = []
            for file in diff.files:
                improvements = self.structured_llm.invoke(
                    self.analysis_prompt.format(diff=file.content)
                )
                
                # Handle single or multiple improvements
                if isinstance(improvements, CodeImprovement):
                    improvements = [improvements]
                
                # Set the file path for each improvement
                for improvement in improvements:
                    improvement.file_path = file.new_file
                
                all_improvements.extend(improvements)
            
            return all_improvements
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            print(f"Full error: {str(e)}")
            return [CodeImprovement(
                file_path="error",
                line_number=0,
                description=f"Failed to analyze improvements: {str(e)}",
                improvement="",
                context=""
            )]

class ConfigurationManager:
    """Manages configuration settings for the script."""
    
    def __init__(self, config_path: str = "~/.aiScriptConfiguration.json"):
        self.config_path = os.path.expanduser(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Configuration file not found at {self.config_path}")
            print("Please create it with format:")
            print("""
{
    "openai": {
        "api_key": "your-api-key-here",
        "model": "gpt-4"
    }
}
""")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Invalid JSON in configuration file at {self.config_path}")
            sys.exit(1)
    
    def get_provider_config(self, provider: str) -> dict:
        """Get configuration for a specific provider."""
        return self.config.get(provider, {})

def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description='Analyze git diffs for code improvements')
    parser.add_argument('commit_id', nargs='?', default='HEAD', help='Commit ID to analyze')
    parser.add_argument('--annotate-pr', action='store_true', help='Output GitHub-style PR annotations')
    args = parser.parse_args()
    
    try:
        # Get OpenAI API key from environment or config
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = "gpt-4"
        
        # If no environment variable, try config file
        if not api_key:
            config_manager = ConfigurationManager()
            config = config_manager.get_provider_config("openai")
            api_key = config.get("api_key")
            model_name = config.get("model", model_name)
            
            if not api_key:
                print("Error: No API key found in environment or config file")
                return

        # Initialize analyzer
        analyzer = DiffAnalyzer(
            api_key=api_key,
            model_name=model_name
        )
        
        # Get the diff
        diff = analyzer.get_git_diff(args.commit_id)
        if not diff:
            print(f"No changes found in commit: {args.commit_id}")
            return
        
        # Analyze the diff
        improvements = analyzer.analyze_diff(diff)
        
        # Display results
        print("\n# Code Improvement Suggestions:")
        print("=" * 80)
        
        for i, improvement in enumerate(improvements, 1):
            if args.annotate_pr:
                # Output GitHub-style annotations
                print(f"::notice file={improvement.file_path},line={improvement.line_number},title=Code Improvement Suggestion::{improvement.description}\n{improvement.improvement}")
            
            # Write to stdout for GitHub summary
            print(f"\n### Suggestion {i}")
            print(f"**File:** {improvement.file_path}")
            print(f"**Line:** {improvement.line_number}")
            print(f"Issue: {improvement.description}")
            print(f"Suggested Improvement: {improvement.improvement}")
            print("---")
            
            if i < len(improvements):
                print("\nPress Enter to see next suggestion...", file=sys.stderr)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 