from dataclasses import dataclass
import subprocess
from typing import List, Optional
from enum import Enum

class LineType(Enum):
    """Enum for different types of lines in a diff."""
    CONTEXT = "context"
    ADDED = "added"
    REMOVED = "removed"

@dataclass
class DiffLine:
    """Represents a single line in a diff."""
    content: str
    line_type: LineType
    
    @classmethod
    def from_diff_line(cls, line: str) -> 'DiffLine':
        """Create a DiffLine from a raw git diff line."""
        if line.startswith('+'):
            return cls(content=line[1:], line_type=LineType.ADDED)
        elif line.startswith('-'):
            return cls(content=line[1:], line_type=LineType.REMOVED)
        else:
            return cls(content=line, line_type=LineType.CONTEXT)

@dataclass
class DiffHunk:
    """Represents a hunk of changes in a diff."""
    lines: List[DiffLine]
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    
    @classmethod
    def from_hunk_header(cls, header: str, lines: List[str]) -> 'DiffHunk':
        """Create a DiffHunk from a hunk header and lines."""
        # Parse @@ -l,s +l,s @@ format
        header_parts = header.split(' ')
        old_range = header_parts[1][1:].split(',')
        new_range = header_parts[2][1:].split(',')
        
        return cls(
            lines=[DiffLine.from_diff_line(line) for line in lines],
            old_start=int(old_range[0]),
            old_count=int(old_range[1]) if len(old_range) > 1 else 1,
            new_start=int(new_range[0]),
            new_count=int(new_range[1]) if len(new_range) > 1 else 1
        )

@dataclass
class DiffFile:
    """Represents a file in a git diff."""
    old_file: str
    new_file: str
    content: str
    has_changes: bool = True

@dataclass
class GitFile:
    """Represents a file in a git diff."""
    new_file: str
    content: str

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