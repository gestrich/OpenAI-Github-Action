from dataclasses import dataclass
import subprocess
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import sys
import json
import os

@dataclass
class GitDiff:
    """Represents a complete git diff."""
    content: str

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
        try:
            if commit_id:
                # Get diff between commit and its parent
                diff_output = subprocess.check_output(
                    ["git", "diff", "-U50", f"{commit_id}^", commit_id],
                    text=True,
                    stderr=subprocess.PIPE
                )
            else:
                # Get diff of staged changes
                diff_output = subprocess.check_output(
                    ["git", "diff", "--cached", "-U50"],
                    text=True,
                    stderr=subprocess.PIPE
                )
            
            return GitDiff(content=diff_output) if diff_output else None
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting git diff: {e}")
            if e.stderr:
                print(f"Git error: {e.stderr.decode()}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def analyze_diff(self, diff: GitDiff) -> List[CodeImprovement]:
        """Analyze the entire diff for improvements."""
        try:
            improvements = self.structured_llm.invoke(
                self.analysis_prompt.format(diff=diff.content)
            )
            
            # Handle single or multiple improvements
            if isinstance(improvements, CodeImprovement):
                return [improvements]
            return improvements
            
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
        
        # Get commit ID from command line or use default
        commit_id = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
        
        # Get the diff
        diff = analyzer.get_git_diff(commit_id)
        if not diff:
            print(f"No changes found in commit: {commit_id}")
            return
        
        # Analyze the diff
        improvements = analyzer.analyze_diff(diff)
        
        # Display results
        print("\nCode Improvement Suggestions:")
        print("=" * 80)
        
        # Write to GitHub step summary
        for i, improvement in enumerate(improvements, 1):
            # Output GitHub-style annotations
            print(f"::notice file={improvement.file_path},line={improvement.line_number},title=Code Improvement Suggestion::{improvement.description}\n{improvement.improvement}")
            
            # Also write to summary
            print(f"\n### Suggestion {i}:", file=sys.stderr)
            print(f"**File:** {improvement.file_path}", file=sys.stderr)
            print(f"**Line:** {improvement.line_number}", file=sys.stderr)
            print("\n**Issue:**", file=sys.stderr)
            print(improvement.description, file=sys.stderr)
            print("\n**Suggested Improvement:**", file=sys.stderr)
            print(improvement.improvement, file=sys.stderr)
            print("---", file=sys.stderr)
            
            if i < len(improvements):
                input("\nPress Enter to see next suggestion...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 