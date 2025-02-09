import subprocess
import sys
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
import json
import os
try:
    import openai
except ImportError:
    print("OpenAI package not found. Please install required dependencies:")
    print("pip install -r requirements.txt")
    # or directly:
    print("pip install openai>=1.0.0")
    sys.exit(1)
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from git_diff import GitDiff, GitFile, get_git_diff

class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        pass

class OllamaProvider(AIProvider):
    """Ollama-specific implementation."""
    
    def __init__(self, model: str = "llama2"):
        self.model = model
    
    def generate_response(self, prompt: str) -> str:
        try:
            cmd = ['ollama', 'run', self.model, prompt]
            result = subprocess.run(cmd, text=True, capture_output=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running ollama: {e}")
            print(f"Error output: {e.stderr}")
            return "Failed to generate response"

class OpenAIProvider(AIProvider):
    """OpenAI-specific implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert code analyzer focusing on Objective-C nullability changes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0  # Keep responses consistent
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return "Failed to generate response"

class ReasoningStepType(Enum):
    """Types of reasoning steps in the analysis chain."""
    CLASSIFICATION = auto()      # Classify the type of change
    ANALYSIS = auto()           # Analyze the implications
    RISK_ASSESSMENT = auto()    # Assess risks
    RECOMMENDATION = auto()     # Make recommendations
    VALIDATION = auto()          # Validate nil propagation
    BEHAVIOR_CHANGE = auto()     # Analyze behavior changes in nil propagation

@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_type: ReasoningStepType
    prompt_template: str
    process_response: Optional[Callable[[str], Any]] = None
    should_execute: Optional[Callable[[dict], bool]] = None
    context: dict = None
    
    def generate_prompt(self, previous_results: dict, chunk: str) -> str:
        """Generate the prompt for this step using previous results."""
        return self.prompt_template.format(
            chunk=chunk,
            **previous_results,
            **(self.context or {})
        )

    def process_result(self, response: str) -> Any:
        """Process the response using the provided function or return as-is."""
        if self.process_response is None:
            return response
        try:
            return self.process_response(response)
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Raw response: {response}")
            return response

class ReasoningChain:
    """A chain of reasoning steps for analyzing code changes."""
    
    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider
        self.steps = []
        self.results = {}
    
    def add_step(self, step: ReasoningStep) -> 'ReasoningChain':
        """Add a step to the chain."""
        self.steps.append(step)
        return self
    
    def format_step_result(self, step: ReasoningStep, result: Any) -> str:
        """Format the step result for display."""
        if step.step_type == ReasoningStepType.CLASSIFICATION:
            return f"""
Classification Type: {self._get_classification_name(result['type'])}

Reasoning:
{result['reasoning']}
"""
        elif step.step_type == ReasoningStepType.BEHAVIOR_CHANGE:
            changes = []
            if result['categories']['api_skipped']:
                changes.append("✓ API requiring non-nil no longer called")
            else:
                changes.append("✗ API requiring non-nil no longer called")
                
            if result['categories']['execution_changed']:
                changes.append("✓ Other execution changed")
            else:
                changes.append("✗ Other execution changed")
                
            return f"""
Behavior Changes:
{chr(10).join(changes)}

Evidence:
{result['evidence']}

Reasoning:
{result['reasoning']}
"""
        else:
            # For other step types, just return the result as is
            return str(result)
    
    def _get_classification_name(self, type_num: int) -> str:
        """Convert classification number to readable name."""
        classifications = {
            1: "Defensive nil check (early return/guard)",
            2: "Null coalescing (using ?: or ??)",
            3: "Nullability annotation change",
            4: "Nil propagation handling",
            5: "Other nullability change"
        }
        return classifications.get(type_num, "Unknown")
    
    def execute(self, chunk: str) -> dict:
        """Execute all steps in the chain."""
        for step in self.steps:
            if step.should_execute and not step.should_execute(self.results):
                continue
                
            print(f"\nExecuting {step.step_type.name} step...")
            print(f"Using provider: {self.ai_provider.__class__.__name__}")
            print(f"Model: {self.ai_provider.model}\n")
            
            prompt = step.generate_prompt(self.results, chunk)
            response = self.ai_provider.generate_response(prompt)
            
            # Use the new process_result method
            self.results[step.step_type.name] = step.process_result(response)
            
            print(f"Step result:")
            print("=" * 80)
            print(self.format_step_result(step, self.results[step.step_type.name]))
            print("=" * 80)
            
            input("Press Enter to continue to next step...")
        
        return self.results

class NullabilityClassification(BaseModel):
    """Pydantic model for nullability classification output."""
    type_number: int = Field(description="Classification type number (1-5)")
    reasoning: str = Field(description="Detailed reasoning for the classification")

class BehaviorChangeCategories(BaseModel):
    """Categories of behavior changes."""
    api_skipped: bool = Field(description="Whether APIs requiring non-nil are no longer called")
    execution_changed: bool = Field(description="Whether other execution paths are changed")

class BehaviorChangeAnalysis(BaseModel):
    """Pydantic model for behavior change analysis."""
    categories: BehaviorChangeCategories
    evidence: str = Field(description="Evidence supporting the behavior changes")
    reasoning: str = Field(description="Detailed reasoning for the behavior changes")

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

class CodeImprovement(BaseModel):
    """Model for code improvement suggestions."""
    line_number: int = Field(description="The line number where the issue occurs")
    description: str = Field(description="Clear description of the issue found")
    improvement: str = Field(description="Specific suggestion on how to improve the code")
    context: str = Field(default="", description="The surrounding code context")

class NullabilityAnalyzer:
    """Main class for analyzing nullability changes using LangChain."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
            openai_api_key=api_key
        )
        self.setup_chains()
    
    def setup_chains(self):
        """Setup the LangChain chains for analysis."""
        # Convert the LLM to structured output
        self.structured_llm = self.llm.with_structured_output(CodeImprovement)
        
        # Create the improvement analysis prompt
        self.improvement_prompt = f"""
{general_code_analysis_tips()}

Analyze the following code changes and identify potential improvements.
Focus on:
1. Code quality issues
2. Potential bugs
3. Performance improvements
4. Best practices

For each issue found, provide:
1. The specific line number
2. A clear description of the issue
3. A concrete suggestion for improvement

Code to analyze:
{{code}}
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
            
            # Parse the diff output
            files = []
            current_file = None
            current_content = []
            
            for line in diff_output.split('\n'):
                if line.startswith('diff --git'):
                    if current_file and current_content:
                        files.append(GitFile(
                            new_file=current_file,
                            content='\n'.join(current_content)
                        ))
                    current_content = []
                    current_file = None
                elif line.startswith('+++ b/'):
                    current_file = line[6:]
                elif current_file:
                    current_content.append(line)
            
            # Add the last file
            if current_file and current_content:
                files.append(GitFile(
                    new_file=current_file,
                    content='\n'.join(current_content)
                ))
            
            return GitDiff(files=files) if files else None
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting git diff: {e}")
            if e.stderr:
                print(f"Git error: {e.stderr.decode()}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def get_context_lines(self, chunk: str, line_number: int, context: int = 5) -> str:
        """Get surrounding context lines for a specific line number."""
        lines = chunk.split('\n')
        start = max(0, line_number - context)
        end = min(len(lines), line_number + context + 1)
        
        result = []
        for i in range(start, end):
            prefix = '> ' if i == line_number else '  '
            result.append(f"{prefix}{i+1:4d}| {lines[i]}")
        
        return '\n'.join(result)

    def analyze_improvements(self, chunk: str) -> List[CodeImprovement]:
        """Analyze code for potential improvements and bugs."""
        try:
            # Create a version of the code with clear change markers
            marked_lines = []
            changed_line_numbers = []
            
            for i, line in enumerate(chunk.split('\n')):
                if line.startswith('+'):
                    marked_lines.append(f"[CHANGED] {line}")
                    changed_line_numbers.append(i)
                elif line.startswith('-'):
                    marked_lines.append(f"[REMOVED] {line}")
                else:
                    marked_lines.append(f"[CONTEXT] {line}")
            
            # Update the prompt to focus on changed lines while showing context
            analysis_prompt = f"""
{general_code_analysis_tips()}

Below is code with context, where each line is marked as either:
- [CHANGED]: New or modified lines
- [REMOVED]: Removed lines
- [CONTEXT]: Unchanged context lines

IMPORTANT: Only analyze and make suggestions for the [CHANGED] lines.
Do not make suggestions about the context or removed lines.
Use the context only to understand the changes better.

Focus on these aspects of the changed code:
1. Code quality issues
2. Potential bugs
3. Performance improvements
4. Best practices

When making suggestions:
- Only refer to line numbers of [CHANGED] lines
- Consider the context but don't suggest changes to it
- Ensure suggestions are specific to the actual changes

Code to analyze:
{{code}}
"""
            
            # Get improvements using structured output
            improvements = self.structured_llm.invoke(
                analysis_prompt.format(code='\n'.join(marked_lines))
            )
            
            # Add context to each improvement
            if isinstance(improvements, CodeImprovement):
                improvements = [improvements]
            
            for improvement in improvements:
                improvement.context = self.get_context_lines(
                    chunk, 
                    improvement.line_number - 1  # Convert to 0-based index
                )
            
            return improvements
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            print(f"Full error: {str(e)}")
            return [CodeImprovement(
                line_number=0,
                description="Failed to analyze improvements",
                improvement="",
                context=""
            )]

    def analyze_chunk(self, chunk: str) -> str:
        """Analyze a chunk of code for nullability changes."""
        try:
            # Create a version of the code with clear change markers
            marked_lines = []
            for line in chunk.split('\n'):
                if line.startswith('+'):
                    marked_lines.append(f"[CHANGED] {line}")
                elif line.startswith('-'):
                    marked_lines.append(f"[REMOVED] {line}")
                else:
                    marked_lines.append(f"[CONTEXT] {line}")
            
            nullability_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
{general_code_analysis_tips()}

Below is code with context, where each line is marked as either:
- [CHANGED]: New or modified lines
- [REMOVED]: Removed lines
- [CONTEXT]: Unchanged context lines

IMPORTANT: Analyze ONLY the [CHANGED] lines for nullability changes.
Use the context only to understand the changes better.

Classify the type of nullability changes in the modified code:

1. Defensive nil check (early return/guard)
   Example: if (x == nil) return;
   
2. Null coalescing (using ?: or ??)
   Example: NSString *text = maybeNil ?: @"default";
   
3. Nullability annotation change
   Example: - (nullable NSString *) or _Nullable changes
   
4. Nil propagation handling
   Example: Changes in how nil flows through method calls
   
5. Other nullability changes
   Any other changes related to nil handling

For each change found:
1. Identify the specific line number
2. Explain what changed
3. Describe the impact on nil safety
"""),
                ("user", "Here is the code to analyze:\n{code}")
            ])
            
            chain = (
                nullability_prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            return chain.invoke({"code": '\n'.join(marked_lines)})
            
        except Exception as e:
            print(f"Error during nullability analysis: {e}")
            print(f"Full error: {str(e)}")
            return f"Failed to analyze nullability: {str(e)}"

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

class RepoAssistant:
    """Assistant for generating repository documentation."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
            openai_api_key=api_key
        )
        self.setup_tools()
    
    @tool
    def get_repo_directories(self, _: str = "") -> list:
        """Get all directories in the iOS project path."""
        try:
            # Use the specific iOS project path
            repo_path = "/Users/bill/Developer/work/ios"
            
            # List all items in the directory
            items = os.listdir(repo_path)
            
            # Filter for directories only, exclude hidden directories
            directories = [
                d for d in items 
                if os.path.isdir(os.path.join(repo_path, d)) 
                and not d.startswith('.')
            ]
            
            # Sort alphabetically for consistent output
            directories.sort()
            
            return directories
        except Exception as e:
            print(f"Error getting directories: {str(e)}")
            return [f"Error: {str(e)}"]
    
    def setup_tools(self):
        """Setup the LangChain tools and prompts."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a repository documentation assistant. Using the provided repository structure,
create a clear and professional README.md content."""),
            ("user", """Repository directories:
{directories}

Create a README that:
1. Has a clear title (derived from the repository structure)
2. Lists and describes the main directories
3. Uses proper markdown formatting
4. Keeps descriptions concise but informative
5. Includes a standard structure (Overview, Directory Structure, etc.)

Please generate the README.md content:""")
        ])
        
        # Create the chain
        self.chain = (
            self.prompt 
            | self.llm.bind_tools([self.get_repo_directories])
            | StrOutputParser()
        )
    
    def generate_readme(self) -> str:
        """Generate README content based on repository structure."""
        try:
            # Get directories using the tool
            directories = self.get_repo_directories("")
            
            # Generate README content using the chain
            result = self.chain.invoke({
                "directories": "\n".join(f"- {d}" for d in directories)
            })
            
            return result
            
        except Exception as e:
            print(f"Full error: {str(e)}")  # Add detailed error logging
            return f"Error generating README: {str(e)}"

class CodeAnalysis(BaseModel):
    """Model for code analysis results."""
    improvements: List[CodeImprovement]

    def analyze_improvements(self, chunk: str) -> List[CodeImprovement]:
        """Analyze code for potential improvements and bugs."""
        improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a code review assistant focusing on code improvements and bug detection.
{general_code_analysis_tips()}

Analyze the code changes and identify potential improvements.
"""),
            ("user", "Here are the code changes to analyze:\n{chunk}")
        ])
        
        try:
            completion = self.llm.beta.chat.completions.parse(
                messages=[
                    improvement_prompt.format_messages(chunk=chunk)
                ],
                response_format=CodeImprovement,
                model="gpt-4"
            )
            
            improvements = completion.choices[0].message.parsed
            
            # Add context to each improvement
            for improvement in improvements:
                improvement.context = self.get_context_lines(
                    chunk, 
                    improvement.line_number - 1  # Convert to 0-based index
                )
            
            return improvements
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            print(f"Full error: {str(e)}")
            return [CodeImprovement(
                line_number=0,
                description="Failed to analyze improvements",
                improvement="",
                context=""
            )]

def main():
    """Entry point for the script."""
    # Load configuration
    config_manager = ConfigurationManager()
    config = config_manager.get_provider_config("openai")
    
    print("\nAvailable operations:")
    print("1. Commit Analysis")
    print("2. Generate Repository README")
    print("0. Exit")
    
    choice = input("\nChoose operation: ").strip()
    
    if choice == "0":
        return
    elif choice == "1":
        print("\nSelect analysis type:")
        print("1. Improvements & Bugs")
        print("2. Nullability Change Analysis")
        
        analysis_type = input("\nChoose analysis type: ").strip()
        
        # Initialize analyzer
        analyzer = NullabilityAnalyzer(
            api_key=config.get("api_key"),
            model_name=config.get("model", "gpt-4")
        )
        
        # Get commit ID from command line or use default
        commit_id = sys.argv[1] if len(sys.argv) > 1 else "faa3bd8"
        
        # Get the diff
        diff = analyzer.get_git_diff(commit_id)
        if not diff or not diff.files:
            print(f"No changes found in commit: {commit_id}")
            return
        
        for file in diff.files:
            print(f"\n=== Analyzing file: {file.new_file} ===\n")
            
            changed_lines = [line for line in file.content.split('\n') 
                           if line.startswith('+') or line.startswith('-')]
            
            if changed_lines:
                print("CHANGED LINES:")
                print("=" * 80)
                for line in changed_lines:
                    print(line)
                print("=" * 80)
                print()
            
            if analysis_type == "1":
                improvements = analyzer.analyze_improvements(file.content)
                print("\nImprovements & Potential Issues:")
                for i, improvement in enumerate(improvements, 1):
                    print(f"\nSuggestion {i}:")
                    print("=" * 80)
                    
                    if improvement.context:
                        print("CONTEXT:")
                        print(improvement.context)
                        print()
                    
                    print("ISSUE:")
                    print(improvement.description)
                    print()
                    
                    print("SUGGESTED IMPROVEMENT:")
                    print(improvement.improvement)
                    print("=" * 80)
                    
                    if i < len(improvements):
                        input("\nPress Enter to see next suggestion...")
            else:
                analysis = analyzer.analyze_chunk(file.content)
                print("\nNullability Analysis:")
                print(analysis)
            
            if len(diff.files) > 1:
                input("\nPress Enter to continue to next file...")
    
    elif choice == "2":
        # Repository documentation generation (unchanged)
        assistant = RepoAssistant(
            api_key=config.get("api_key"),
            model_name=config.get("model", "gpt-4")
        )
        
        print("\nAnalyzing repository structure...")
        readme_content = assistant.generate_readme()
        
        print("\nGenerated README.md content:")
        print("=" * 80)
        print(readme_content)
        print("=" * 80)
        
        save = input("\nWould you like to save this as README.md? (y/n): ").strip().lower()
        if save == 'y':
            try:
                with open("README.md", "w") as f:
                    f.write(readme_content)
                print("README.md has been saved!")
            except Exception as e:
                print(f"Error saving README.md: {e}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()

