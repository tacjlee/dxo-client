"""
Golden Test Case Parser

Parses golden test case markdown files (exported from Excel) into structured data
that can be sent to the graph knowledge service.

File format:
- Markdown tables exported from Excel test case sheets
- Header row contains: ID | Viewpoint | Recommend | Test Item | Pre-Condition | ...
- Data rows contain test cases with viewpoint assignments
"""

import re
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

from ..models.graph_knowledge import GoldenTestCase, GoldenWidget

logger = logging.getLogger(__name__)


@dataclass
class GoldenFileData:
    """Parsed golden file data ready for API submission."""
    screen_id: str
    screen_name: str
    mode: str
    module_code: str
    document_type: str = "CRUD"
    testcases: List[GoldenTestCase] = field(default_factory=list)
    widgets: List[GoldenWidget] = field(default_factory=list)


def _parse_markdown_table_row(line: str) -> List[str]:
    """Parse a markdown table row into cells."""
    if not line.strip().startswith('|'):
        return []
    cells = line.split('|')
    if cells and cells[0].strip() == '':
        cells = cells[1:]
    if cells and cells[-1].strip() == '':
        cells = cells[:-1]
    return [cell.strip() for cell in cells]


def _extract_screen_info(filename: str) -> tuple[str, str, str]:
    """
    Extract screen ID, name, and mode from filename.

    Examples:
        SC011_AddAccount.md -> (SC011, AddAccount, Add)
        SC011_EditAccount.md -> (SC011, EditAccount, Edit)
        SC011_ListAccount.md -> (SC011, ListAccount, List)
        SC004_RequestDetail.md -> (SC004, RequestDetail, Detail)
    """
    name = Path(filename).stem

    # Extract screen ID (SCXXX pattern)
    screen_match = re.match(r'^(SC\d+)', name)
    screen_id = screen_match.group(1) if screen_match else name.split('_')[0]

    # Extract mode from name
    mode = "Unknown"
    name_lower = name.lower()
    if 'add' in name_lower:
        mode = "Add"
    elif 'edit' in name_lower:
        mode = "Edit"
    elif 'list' in name_lower:
        mode = "List"
    elif 'detail' in name_lower:
        mode = "Detail"
    elif 'delete' in name_lower:
        mode = "Delete"
    elif 'search' in name_lower:
        mode = "Search"

    # Screen name is the part after screen ID
    screen_name = name.replace(screen_id + '_', '') if '_' in name else name

    return screen_id, screen_name, mode


def _is_section_header(row: List[str]) -> bool:
    """Check if a row is a section header (like 'Validate', 'Function', 'GUI')."""
    if len(row) < 2:
        return False
    first_cell = row[0].strip()
    if not first_cell:
        return False
    # Section headers have text in first column but empty or minimal in other columns
    non_empty_count = sum(1 for cell in row[1:8] if cell.strip())
    # Not a formula and most cells empty
    return non_empty_count <= 1 and not first_cell.startswith('=')


def _is_data_row(row: List[str]) -> bool:
    """Check if a row contains test case data."""
    if len(row) < 8:
        return False
    # Must have procedure (col 5) or expected output (col 7)
    procedure = row[5].strip() if len(row) > 5 else ''
    expected = row[7].strip() if len(row) > 7 else ''
    return bool(procedure or expected)


def _is_header_row(row: List[str]) -> bool:
    """Check if this is the header row."""
    if len(row) < 5:
        return False
    row_text = ' '.join(row[:10]).lower()
    return 'viewpoint' in row_text and ('test item' in row_text or 'procedure' in row_text)


def parse_golden_file(file_path: str) -> GoldenFileData:
    """
    Parse a golden test case markdown file.

    Args:
        file_path: Path to the golden markdown file

    Returns:
        GoldenFileData with parsed test cases

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden file not found: {file_path}")

    content = path.read_text(encoding='utf-8')
    lines = content.split('\n')

    screen_id, screen_name, mode = _extract_screen_info(path.name)

    golden_data = GoldenFileData(
        screen_id=screen_id,
        screen_name=screen_name,
        mode=mode,
        module_code='',
    )

    current_section = None
    current_viewpoint = None
    header_found = False
    testcase_count = 0

    for line in lines:
        row = _parse_markdown_table_row(line)

        if not row:
            continue

        # Skip separator rows (---)
        if row[0].startswith('---') or all(c == '-' or c == '' for c in row[0]):
            continue

        # Extract module code from first data row
        if len(row) > 1 and row[0] == 'Module Code':
            golden_data.module_code = row[1].strip()
            continue

        # Find header row
        if not header_found and _is_header_row(row):
            header_found = True
            continue

        if not header_found:
            continue

        # Skip sub-header row (Result, Bug ID/Link, etc.)
        if len(row) > 12 and 'Result' in row[12]:
            continue

        # Check for section header
        if _is_section_header(row):
            current_section = row[0].strip()
            logger.debug(f"  Section: {current_section}")
            continue

        # Parse data row
        if _is_data_row(row):
            # Extract viewpoint - may be empty if continuation of previous
            viewpoint = row[1].strip() if len(row) > 1 else ''
            if viewpoint:
                current_viewpoint = viewpoint

            # Skip if no viewpoint context
            if not current_viewpoint:
                continue

            testcase_count += 1

            # Build test case content from available fields
            test_item = row[3].strip() if len(row) > 3 else ''
            recommend = row[2].strip() if len(row) > 2 else ''
            content = test_item or recommend or f"Test case {testcase_count}"

            testcase = GoldenTestCase(
                content=content,
                procedure=row[5].strip() if len(row) > 5 else '',
                expected_result=row[7].strip() if len(row) > 7 else '',
                viewpoint=current_viewpoint,
                mode=mode,
                priority=row[8].strip() if len(row) > 8 else 'Medium',
            )

            golden_data.testcases.append(testcase)

    logger.info(f"Parsed {len(golden_data.testcases)} test cases from {path.name}")
    return golden_data


def parse_golden_directory(directory: str) -> List[GoldenFileData]:
    """
    Parse all golden files in a directory.

    Args:
        directory: Path to directory containing golden markdown files

    Returns:
        List of GoldenFileData for each file
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    results = []
    for md_file in sorted(dir_path.glob('*.md')):
        try:
            data = parse_golden_file(str(md_file))
            if data.testcases:
                results.append(data)
        except Exception as e:
            logger.error(f"Error parsing {md_file.name}: {e}")

    return results
