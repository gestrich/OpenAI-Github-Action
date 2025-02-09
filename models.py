from enum import Enum
from dataclasses import dataclass
from typing import List

class LineType(Enum):
    """Type of line in a diff."""
    ADDED = "added"
    REMOVED = "removed"
    CONTEXT = "context"

@dataclass(frozen=True)  # Make the class immutable and hashable
class Line:
    """Represents a line in the code with its metadata."""
    content: str
    number: int
    type: LineType
    
    def __hash__(self):
        """Make Line hashable for use in sets."""
        return hash((self.content, self.number, self.type))
    
    @classmethod
    def from_diff_line(cls, content: str, number: int) -> 'Line':
        """Create a Line instance from a git diff line."""
        if content.startswith('+'):
            return cls(content=content[1:], number=number, type=LineType.ADDED)
        elif content.startswith('-'):
            return cls(content=content[1:], number=number, type=LineType.REMOVED)
        else:
            return cls(content=content, number=number, type=LineType.CONTEXT)

    def to_dict(self) -> dict:
        """Convert to dictionary for API consumption."""
        return {
            "content": self.content,
            "line_number": self.number,
            "type": self.type.value
        }
    
    def format_line(self, line_number_width: int = 4) -> str:
        """Format the line with line number and change indicator.
        
        Args:
            line_number_width: Width to use for line number column
            
        Returns:
            Formatted string like: "  12 | + some code"
        """
        change_indicator = {
            LineType.ADDED: "+",
            LineType.REMOVED: "-",
            LineType.CONTEXT: " "
        }[self.type]
        
        return f"{str(self.number).rjust(line_number_width)} | {change_indicator} {self.content}"

    @staticmethod
    def format_lines(lines: List['Line']) -> str:
        """Format a list of lines with aligned columns.
        
        Returns:
            Multi-line string with formatted lines
        """
        if not lines:
            return ""
        
        # Calculate the width needed for line numbers
        max_line_number = max(line.number for line in lines)
        line_number_width = len(str(max_line_number))
        
        # Format each line
        return "\n".join(
            line.format_line(line_number_width)
            for line in lines
        ) 