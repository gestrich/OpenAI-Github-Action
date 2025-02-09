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
from enum import Enum
from models import Line, LineType

class CodeImprovement(BaseModel):
    """Model for code quality issues and suggested fixes."""
    file_path: str = Field(
        description="The file path where the potential issue was found"
    )
    line_number: int = Field(
        description="The line number where the potential issue exists"
    )
    potential_bug: str = Field(
        description="Describe a specific code quality issue, bug risk, or potential problem in the ADDED code. " +
        "DO NOT describe what the code does or what changed - only describe problems or risks."
    )
    suggested_fix: str = Field(
        description="Provide a specific, actionable suggestion for how to fix the potential issue. " +
        "DO NOT describe what the code currently does - only describe how to improve it."
    )
    context_lines: List[Line] = Field(
        default_factory=list,
        description="The surrounding code context as Line objects"
    )

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

You are a code reviewer looking for potential issues in new code changes.
Your task is to ONLY analyze the ADDED lines (type: "added") for potential problems.

DO NOT:
- Describe what the code does
- Explain the changes made
- Summarize the commit
- Comment on removed code
- Describe the implementation

INSTEAD, look for these specific issues in ADDED lines:
1. Potential bugs or edge cases that could cause failures
2. Security vulnerabilities or risks
3. Performance problems or inefficiencies
4. Code quality issues or anti-patterns
5. Maintainability concerns
6. Best practice violations

For each issue found, you must:
1. potential_bug: Describe a specific problem or risk (NOT what the code does)
2. suggested_fix: Provide specific code changes to fix the issue (NOT what changed)

Example GOOD response:
- potential_bug: "The function doesn't validate the input parameter, risking null pointer exceptions"
- suggested_fix: "Add input validation: if (param == null) throw new IllegalArgumentException()"

Example BAD response:
- potential_bug: "The code was updated to handle line numbers"
- suggested_fix: "The changes look good and improve the output format"

Lines to analyze:
{{lines}}
"""

    def get_git_diff(self, commit_id: Optional[str] = None) -> Optional[GitDiff]:
        """Get the git diff for analysis."""
        return get_git_diff(commit_id)

    def analyze_diff(self, diff: GitDiff, annotate_pr: bool = False) -> List[CodeImprovement]:
        """Analyze the entire diff for improvements."""
        try:
            all_improvements = []
            
            # Analyze each file separately
            for file in diff.files:
                # Convert lines to dict for API consumption
                lines_dict = [line.to_dict() for line in file.lines]
                
                # Debug: Print payload size information
                payload = self.analysis_prompt.format(lines=json.dumps(lines_dict, indent=2))
                payload_size = len(payload.encode('utf-8'))
                print(f"\nDebug - Payload Information for {file.new_file}:")
                print(f"Number of lines: {len(file.lines)}")
                print(f"Payload size: {payload_size / 1024:.2f}KB")
                if payload_size > 100000:  # Warning if over 100KB
                    print(f"WARNING: Large payload size might exceed context limits!")
                
                # Create a new LLM instance for each file to ensure clean context
                file_llm = ChatOpenAI(
                    temperature=0,
                    model_name=self.llm.model_name,
                    openai_api_key=self.llm.openai_api_key
                ).with_structured_output(
                    CodeImprovement,
                    method="function_calling"
                )
                
                # Get improvements from OpenAI for this file
                improvements = file_llm.invoke(
                    payload
                )
                
                # Handle single or multiple improvements
                if isinstance(improvements, CodeImprovement):
                    improvements = [improvements]
                
                # Set the file path and extract context for each improvement
                for improvement in improvements:
                    improvement.file_path = file.new_file
                    
                    # Find the line in our structured format
                    target_line = next(
                        (line for line in file.lines if line.number == improvement.line_number),
                        None
                    )
                    
                    if target_line:
                        # Get surrounding context
                        line_index = file.lines.index(target_line)
                        start = max(0, line_index - 10)
                        end = min(len(file.lines), line_index + 10)
                        
                        # Extract the context lines
                        context_lines = file.lines[start:end]
                        improvement.context_lines = context_lines
                    else:
                        print(f"Warning: Could not find line {improvement.line_number}")
                        improvement.context_lines = []
                
                all_improvements.extend(improvements)
            
            # Print the complete summary at the end
            if all_improvements:
                print("\n# Code Improvement Suggestions:")
                print("=" * 80)
                
                for i, improvement in enumerate(all_improvements, 1):
                    if annotate_pr:
                        # Output GitHub-style annotations
                        print(f"::notice file={improvement.file_path},line={improvement.line_number},title=Code Improvement Suggestion::{improvement.potential_bug}\n{improvement.suggested_fix}")
                    
                    # Write to stdout for GitHub summary
                    print(f"### Suggestion {i}")
                    print(f"**File:** {improvement.file_path}")
                    print(f"**Line:** {improvement.line_number}\n")
                    print("**Potential Bug:**")
                    print(f"{improvement.potential_bug}\n")
                    print("**Suggested Fix:**")
                    print(f"```\n{improvement.suggested_fix}\n```")

                    # Show the relevant code context
                    if improvement.context_lines:
                        print("**Relevant Code:**\n")
                        print("```")
                        print(Line.format_lines(improvement.context_lines))
                        print("```")
                    print("---")
            
            return all_improvements
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            print(f"Full error: {str(e)}")
            return [CodeImprovement(
                file_path="error",
                line_number=0,
                description=f"Failed to analyze improvements: {str(e)}",
                improvement="",
                context_lines=[]
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
        improvements = analyzer.analyze_diff(diff, annotate_pr=args.annotate_pr)
        
        if not improvements:
            print("No improvements suggested.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 