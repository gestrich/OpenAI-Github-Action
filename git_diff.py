from dataclasses import dataclass
import subprocess
import re
from typing import List, Optional
from models import Line, LineType

@dataclass
class GitFile:
    """Represents a file in a git diff."""
    new_file: str
    lines: List[Line]
    
    def to_text(self) -> str:
        """Convert lines back to text format."""
        return '\n'.join(line.content for line in self.lines)

@dataclass
class GitDiff:
    """Represents a git diff."""
    files: List[GitFile]

def get_git_diff(commit_id: Optional[str] = None) -> Optional[GitDiff]:
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
                ["git", "diff", "HEAD", "-U50"],
                text=True,
                stderr=subprocess.PIPE
            )
        
        # Parse the diff output
        files = []
        current_file = None
        current_lines = []
        line_number = 1  # Will be updated by @@ headers
        
        # Regular expression to parse @@ headers
        hunk_header_pattern = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@')
        
        for line in diff_output.split('\n'):
            if line.startswith('diff --git'):
                if current_file and current_lines:
                    files.append(GitFile(
                        new_file=current_file,
                        lines=current_lines
                    ))
                current_lines = []
                current_file = None
            elif line.startswith('+++ b/'):
                current_file = line[6:]
            elif line.startswith('@@'):
                # Parse the hunk header to get the starting line number
                match = hunk_header_pattern.match(line)
                if match:
                    line_number = int(match.group(1))
            elif current_file and not line.startswith('+++') and not line.startswith('---'):
                if not line.startswith('@@'):  # Skip @@ lines
                    # Create Line object and track line numbers
                    diff_line = Line.from_diff_line(line, line_number)
                    current_lines.append(diff_line)
                    if diff_line.type != LineType.REMOVED:  # Don't increment line number for removed lines
                        line_number += 1
        
        # Add the last file
        if current_file and current_lines:
            files.append(GitFile(
                new_file=current_file,
                lines=current_lines
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