"""Batch converter for Mathematica .m files to Python.

Transforms Wolfram Language syntax to Python syntax for WikiKG knowledge graphs.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterator


# Lines to strip (package declarations)
STRIP_PATTERNS = [
    r'^\s*BeginPackage\s*\[.*\]\s*;?\s*$',
    r'^\s*Begin\s*\[.*\]\s*;?\s*$',
    r'^\s*End\s*\[\s*\]\s*;?\s*$',
    r'^\s*EndPackage\s*\[\s*\]\s*;?\s*$',
]

# Import header to prepend
IMPORT_HEADER = "from wikikg import *\n\n"

# Unicode character replacements for Python compatibility
UNICODE_REPLACEMENTS = {
    # Curly/smart quotes -> straight quotes
    '"': '"',  # U+201C LEFT DOUBLE QUOTATION MARK
    '"': '"',  # U+201D RIGHT DOUBLE QUOTATION MARK
    '„': '"',  # U+201E DOUBLE LOW-9 QUOTATION MARK
    '«': '"',  # U+00AB LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '»': '"',  # U+00BB RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    ''': "'",  # U+2018 LEFT SINGLE QUOTATION MARK
    ''': "'",  # U+2019 RIGHT SINGLE QUOTATION MARK
    '‚': "'",  # U+201A SINGLE LOW-9 QUOTATION MARK
    '‹': "'",  # U+2039 SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '›': "'",  # U+203A SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '′': "'",  # U+2032 PRIME (arcminutes)
    '″': "''",  # U+2033 DOUBLE PRIME (arcseconds) - use '' to avoid breaking strings
    # Dashes -> hyphen-minus
    '–': '-',  # U+2013 EN DASH
    '—': '-',  # U+2014 EM DASH
    '−': '-',  # U+2212 MINUS SIGN
    '‐': '-',  # U+2010 HYPHEN
    '‑': '-',  # U+2011 NON-BREAKING HYPHEN
    # Spaces
    '\u00A0': ' ',  # NO-BREAK SPACE
    '\u2003': ' ',  # EM SPACE
    '\u2002': ' ',  # EN SPACE
    '\u2009': ' ',  # THIN SPACE
    # Other punctuation
    '…': '...',  # U+2026 HORIZONTAL ELLIPSIS
    '·': '.',  # U+00B7 MIDDLE DOT
    # Symbols that cause issues in strings (escape them)
    '°': ' degrees ',  # U+00B0 DEGREE SIGN -> word to avoid syntax issues
    '×': 'x',  # U+00D7 MULTIPLICATION SIGN
    '÷': '/',  # U+00F7 DIVISION SIGN
    '±': '+/-',  # U+00B1 PLUS-MINUS SIGN
    '≈': '~',  # U+2248 ALMOST EQUAL TO
    '≠': '!=',  # U+2260 NOT EQUAL TO
    '≤': '<=',  # U+2264 LESS-THAN OR EQUAL TO
    '≥': '>=',  # U+2265 GREATER-THAN OR EQUAL TO
    '→': '->',  # U+2192 RIGHTWARDS ARROW
    '←': '<-',  # U+2190 LEFTWARDS ARROW
    '↔': '<->',  # U+2194 LEFT RIGHT ARROW
}


def normalize_unicode(text: str) -> str:
    """Replace problematic Unicode characters with ASCII equivalents.

    This handles curly quotes, smart quotes, various dashes, and other
    Unicode characters that cause Python syntax errors.
    """
    for unicode_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, replacement)
    return text


def convert_comments(text: str) -> str:
    """Convert Mathematica comments (* ... *) to Python comments #.

    Handles both single-line and multi-line comments.
    Multi-line comments are converted to multiple Python comment lines.
    """
    def replace_comment(match: re.Match) -> str:
        comment_text = match.group(1).strip()
        # Split by newlines and prefix each line with #
        lines = comment_text.split('\n')
        # Clean up each line and add # prefix
        result_lines = []
        for line in lines:
            cleaned = line.strip()
            if cleaned:
                result_lines.append(f'# {cleaned}')
        return '\n'.join(result_lines) if result_lines else ''

    # Match multi-line comments with DOTALL flag
    text = re.sub(r'\(\*\s*(.*?)\s*\*\)', replace_comment, text, flags=re.DOTALL)
    return text


def convert_brackets(text: str) -> str:
    """Convert function call brackets [ ] to ( ) outside strings.

    This is tricky because we need to preserve brackets inside strings.
    """
    result = []
    in_string = False
    string_char = None
    i = 0

    while i < len(text):
        char = text[i]

        # Handle string boundaries
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                # Check for escaped quote
                if i > 0 and text[i-1] != '\\':
                    in_string = False
                    string_char = None

        # Convert brackets outside strings
        if not in_string:
            if char == '[':
                result.append('(')
            elif char == ']':
                result.append(')')
            else:
                result.append(char)
        else:
            result.append(char)

        i += 1

    return ''.join(result)


def convert_associations(text: str) -> str:
    """Convert Mathematica associations <| ... |> to Python dicts { ... }."""
    # Simple replacement - works for non-nested cases
    text = text.replace('<|', '{')
    text = text.replace('|>', '}')
    return text


def convert_rules(text: str) -> str:
    """Convert Mathematica rules -> to Python dict : (outside strings).

    Be careful with -> that appears in strings.
    """
    result = []
    in_string = False
    string_char = None
    i = 0

    while i < len(text):
        char = text[i]

        # Handle string boundaries
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                if i > 0 and text[i-1] != '\\':
                    in_string = False
                    string_char = None

        # Convert -> to : outside strings
        if not in_string and i < len(text) - 1:
            if text[i:i+2] == '->':
                result.append(': ')
                i += 2
                continue

        result.append(char)
        i += 1

    return ''.join(result)


def convert_add_entity(text: str) -> str:
    """Convert AddEntity[Name, ...] to Name = AddEntity("Name", ...).

    Captures the first argument (entity name) and creates an assignment.
    """
    # Pattern: AddEntity( followed by an unquoted identifier
    pattern = r'AddEntity\(\s*([A-Z][A-Za-z0-9_]*)\s*,'

    def replace_add_entity(match: re.Match) -> str:
        name = match.group(1)
        return f'{name} = AddEntity("{name}",'

    return re.sub(pattern, replace_add_entity, text)


def strip_package_lines(lines: list[str]) -> list[str]:
    """Remove package declaration lines."""
    result = []
    for line in lines:
        should_strip = False
        for pattern in STRIP_PATTERNS:
            if re.match(pattern, line):
                should_strip = True
                break
        if not should_strip:
            result.append(line)
    return result


def convert_line(line: str) -> str:
    """Apply all conversions to a single line."""
    line = convert_comments(line)
    line = convert_associations(line)
    line = convert_brackets(line)
    line = convert_rules(line)
    line = convert_add_entity(line)
    # Remove trailing semicolons (Mathematica statement terminators)
    line = re.sub(r';\s*$', '', line)
    return line


def fix_leading_zeros(text: str) -> str:
    """Fix leading zeros in integer literals outside of strings.

    Python 3 doesn't allow leading zeros in decimal integers (e.g., 09 is invalid).
    This converts patterns like 09 to 9, but only outside of quoted strings.
    """
    result = []
    in_string = False
    string_char = None
    i = 0

    while i < len(text):
        char = text[i]

        # Handle string boundaries
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char and (i == 0 or text[i-1] != '\\'):
                in_string = False
                string_char = None
            result.append(char)
            i += 1
            continue

        # Outside strings, fix leading zeros
        if not in_string and char == '0' and i + 1 < len(text):
            next_char = text[i + 1]
            # Check if this is a leading zero followed by more digits (not 0x, 0o, 0b, or decimal point)
            if next_char.isdigit() and next_char != '0':
                # Check that it's not preceded by a digit (part of a larger number)
                if i == 0 or not text[i - 1].isdigit():
                    # Skip the leading zero
                    i += 1
                    continue

        result.append(char)
        i += 1

    return ''.join(result)


def convert_line_no_comments(line: str) -> str:
    """Apply all conversions except comment conversion (for when comments are handled separately)."""
    line = convert_associations(line)
    line = convert_brackets(line)
    line = convert_rules(line)
    line = convert_add_entity(line)
    line = fix_leading_zeros(line)
    # Remove trailing semicolons (Mathematica statement terminators)
    line = re.sub(r';\s*$', '', line)
    return line


def convert_mathematica_to_python(m_code: str) -> str:
    """Convert Mathematica code to Python.

    Args:
        m_code: Mathematica source code

    Returns:
        Converted Python code
    """
    # First, normalize Unicode characters (curly quotes, dashes, etc.)
    m_code = normalize_unicode(m_code)

    # Convert multi-line comments BEFORE splitting into lines
    m_code = convert_comments(m_code)

    lines = m_code.split('\n')

    # Strip package declarations
    lines = strip_package_lines(lines)

    # Convert each line (comments already handled above)
    converted_lines = [convert_line_no_comments(line) for line in lines]

    # Remove empty lines at start/end
    while converted_lines and not converted_lines[0].strip():
        converted_lines.pop(0)
    while converted_lines and not converted_lines[-1].strip():
        converted_lines.pop()

    # Add import header
    result = IMPORT_HEADER + '\n'.join(converted_lines)

    return result


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> str:
    """Convert a .m file to .py.

    Args:
        input_path: Path to input .m file
        output_path: Path for output .py file (default: same name with .py extension)

    Returns:
        Path to output file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix('.py')
    else:
        output_path = Path(output_path)

    m_code = input_path.read_text(encoding='utf-8')
    py_code = convert_mathematica_to_python(m_code)
    output_path.write_text(py_code, encoding='utf-8')

    return str(output_path)


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "*.m"
) -> Iterator[tuple[str, str]]:
    """Convert all .m files in a directory.

    Args:
        input_dir: Input directory containing .m files
        output_dir: Output directory for .py files (default: same as input)
        pattern: Glob pattern for finding files

    Yields:
        Tuples of (input_path, output_path) for each converted file
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for m_file in input_dir.glob(pattern):
        py_file = output_dir / m_file.with_suffix('.py').name
        convert_file(m_file, py_file)
        yield str(m_file), str(py_file)


def convert_string_to_file(
    m_code: str,
    output_path: str | Path,
    article_name: str | None = None
) -> str:
    """Convert Mathematica code string to a Python file.

    Useful for processing dataset entries directly.

    Args:
        m_code: Mathematica source code string
        output_path: Path for output .py file
        article_name: Optional article name to add as comment

    Returns:
        Path to output file
    """
    output_path = Path(output_path)

    py_code = convert_mathematica_to_python(m_code)

    if article_name:
        header_comment = f"# Knowledge graph for: {article_name}\n"
        py_code = header_comment + py_code

    output_path.write_text(py_code, encoding='utf-8')

    return str(output_path)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Mathematica .m files to Python"
    )
    parser.add_argument("input", help="Input .m file or directory")
    parser.add_argument("-o", "--output", help="Output .py file or directory")
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively process directories"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        output = convert_file(input_path, args.output)
        print(f"Converted: {input_path} -> {output}")
    elif input_path.is_dir():
        pattern = "**/*.m" if args.recursive else "*.m"
        for inp, out in convert_directory(input_path, args.output, pattern):
            print(f"Converted: {inp} -> {out}")
    else:
        print(f"Error: {input_path} not found")
        exit(1)
