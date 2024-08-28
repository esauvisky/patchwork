
import json
import os
import re
from loguru import logger
import difflib

def normalize_repo_path(working_dir):
    normalized_repo_path = os.path.abspath(working_dir).rstrip('/').lstrip('/')
    return normalized_repo_path

def replace_repo_paths_in_patch(patch, normalized_repo_path):
    escaped_repo_base_path = re.escape(normalized_repo_path)
    new_patch = re.sub(escaped_repo_base_path + "/", "", patch, flags=re.MULTILINE)

    normalized_repo_path_segments = normalized_repo_path.split('/')
    for i in range(len(normalized_repo_path_segments), 0, -1):
        partial_path = "/".join(normalized_repo_path_segments[:i])
        escaped_partial_path = re.escape(partial_path)
        new_patch = re.sub(rf"{escaped_partial_path}/", "", new_patch, flags=re.MULTILINE)

    if new_patch != patch:
        logger.warning(f"Patch filepaths were modified. Stripped the following from the patch file:\n{escaped_repo_base_path}")

    return new_patch

def fix_syntax_errors_in_patch(patch):
    match = re.search(r"^ (?=(?:[-+@]|diff --git|index ))", patch, flags=re.MULTILINE)
    if match:
        patch = re.sub(r"^ (?=([-+@]|diff --git|index ))", r"\1", patch, flags=re.MULTILINE)
        logger.warning("Patch had special lines with leading spaces. Stripped them from the patch file.")
        logger.debug(f"Match: {' '.join(match.groups())}")

    return patch

def filter_and_reconstruct_patch_hunks(raw_patch):
    header, _, initial_hunk_section = raw_patch.partition('\n@@')
    initial_hunk_section = '\n@@' + initial_hunk_section
    hunks = re.split(r'(\n@@.*?@@\n)', initial_hunk_section, flags=re.DOTALL)
    filtered_hunks = [header]

    for i in range(1, len(hunks), 2):
        hunk_header = hunks[i]
        hunk_body = hunks[i+1]
        if not re.search(r'^[+-]', hunk_body, flags=re.MULTILINE):
            logger.warning(f"Hunk {i} does not contain any changes. Skipping it.")
        else:
            filtered_hunks.append(hunk_header + hunk_body)

    return '\n'.join(filtered_hunks)

def append_spaces_to_lines(patch):
    match = re.search(r"^(?!(?:[-+@ ]|diff --git|index |\n))(.*)$", patch, flags=re.MULTILINE)
    if match and not re.search(r"^\s*$", match.group(1)):
        logger.warning(f"Patch has context lines without leading space. Added a space to them")
        logger.debug(f"Match: {' '.join(match.groups())}")
        patch = re.sub(r"^(?!(?:[-+@ ]|diff --git|index |\n))(.*)$", r" \1", patch, flags=re.MULTILINE)

    return patch

def strip_whitespace_from_change_lines(patch):
    if re.search(r"^[+-]\s+$", patch, flags=re.MULTILINE):
        patch = re.sub(r"^(\s*[+-])\s+$", r"\1", patch, flags=re.MULTILINE)
        logger.warning(f"Patch had changes only in whitespace. Stripped them from the patch file.")

    return patch

def replace_line_numbers_in_hunk_headers(patch):
    if re.search(r"^@@ [-+\d\,\s]+ @@", patch, flags=re.MULTILINE):
        patch = re.sub(r"^@@ [-+\d\,\s]+ @@", r"@@ -0,0 +0,0 @@", patch, flags=re.MULTILINE)
        logger.warning(f"Patch had hunk headers with line numbers. Replaced them with @@ -0,0 +0,0 @@.")

    return patch
import re

def split_hunks_into_list(patch):
    # Split the remaining patch into hunks
    hunks = re.split(r'\n@@', patch, flags=re.DOTALL)

    # Combine headers with their corresponding bodies
    hunk_list = [hunks[0]]
    for hunk in hunks[1:]:
        hunk_list.append('@@' + hunk)

    return hunk_list

def ensure_leading_space_in_hunks(hunk_list):
    corrected_hunk_list = []

    for hunk in hunk_list:
        lines = hunk.split("\n")
        corrected_lines = []
        has_changes = False
        for idx, line in enumerate(lines):
            if line == '' or line.startswith((' ', '+', '-', '@')):
              if line.startswith(('+', '-')):
                has_changes = True
            else:
                logger.warning(f"Line {idx} does not start with a valid line prefix. Adding a space:")
                print(line)
                line = ' ' + line
            corrected_lines.append(line)

        if not has_changes:
            logger.warning(f"Hunk does not contain any changes. Skipping it.")
            continue

        corrected_hunk_list.append('\n'.join(corrected_lines))

    return corrected_hunk_list

def prepare_patch_for_git(raw_patch, working_dir):
    normalized_repo_path = normalize_repo_path(working_dir)
    patch = replace_repo_paths_in_patch(raw_patch, normalized_repo_path)
    patch = fix_syntax_errors_in_patch(patch)
    patch = filter_and_reconstruct_patch_hunks(patch)

    hunk_list = split_hunks_into_list(patch)
    corrected_hunk_list = ensure_leading_space_in_hunks(hunk_list)
    patch = ''.join(corrected_hunk_list)

    patch = append_spaces_to_lines(patch)
    patch = strip_whitespace_from_change_lines(patch)
    patch = replace_line_numbers_in_hunk_headers(patch)

    # Final adjustments
    patch = patch.strip("\n") + "\n"

    return patch


def colored_diff(expected, actual):
    # Generate diff
    diff = difflib.unified_diff(expected.split('\n'), actual.split('\n'), lineterm='\n', fromfile='Expected', tofile='Actual')
    # Color the diff1
    return ''.join(
        '\033[91;1m' + line + '\n' if line[0] == '-' else
        '\033[92;1m' + line + '\n' if line[0] == '+' else
        '\033[0;0m' + line + '\n'
        for line in diff)

def test_prepare_patch_for_git(file1, file2):
    input_patch = open(file1, "r").read()
    expected_output = open(file2, "r").read()
    result = prepare_patch_for_git(input_patch, "/fake/working/dir")
    if (result != expected_output):
        logger.error(f"Test failed. Input patch: {file1}. Expected output: {file2}.")
        print(colored_diff(expected_output, result))
        return False

if __name__ == "__main__":
    result = test_prepare_patch_for_git("tests/1_input.patch", "tests/1_expected.patch")
    print(result)

    input_2 = json.loads(open("tests/2_input.json", "r").read())
    result = test_prepare_patch_for_git(input_2, "tests/2_expected.patch")
    print(result)
