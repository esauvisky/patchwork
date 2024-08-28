import json
import re
import os
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

def filter_and_reconstruct_patch_hunks(patch):
    lines = patch.splitlines()
    header_lines = []
    hunks = []
    current_hunk = []

    in_hunk = False
    for line in lines:
        if line.startswith('@@'):
            if current_hunk:
                hunks.append('\n'.join(current_hunk))
                current_hunk = []
            in_hunk = True

        if in_hunk:
            current_hunk.append(line)
        else:
            header_lines.append(line)

    if current_hunk:
        hunks.append('\n'.join(current_hunk))

    filtered_hunks = []
    for hunk in hunks:
        if re.search(r'^[+-]', hunk, flags=re.MULTILINE):
            filtered_hunks.append(hunk)
        else:
            logger.warning(f"Hunk does not contain any changes. Skipping it:\n{hunk}")

    return '\n'.join(header_lines + filtered_hunks)

def append_spaces_to_lines(patch):
    lines = patch.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith((' ', '+', '-', '@', 'diff --git', 'index')) and line.strip():
            logger.warning(f"Line {i + 1} in patch does not start with a valid prefix. Adding a space.")
            lines[i] = ' ' + line

    return '\n'.join(lines)

def strip_whitespace_from_change_lines(patch):
    lines = patch.splitlines()
    for i, line in enumerate(lines):
        if re.match(r'^[+-]\s+$', line):
            lines[i] = line[0]  # Keep only the '+' or '-' sign
            logger.warning(f"Patch had changes only in whitespace at line {i + 1}. Stripped them from the patch file.")

    return '\n'.join(lines)

def replace_line_numbers_in_hunk_headers(patch):
    patch = re.sub(r"^@@ [-+\d\,\s]+ @@", r"@@ -0,0 +0,0 @@", patch, flags=re.MULTILINE)
    if "@@ -0,0 +0,0 @@" in patch:
        logger.warning(f"Patch had hunk headers with line numbers. Replaced them with @@ -0,0 +0,0 @@.")

    return patch

def prepare_patch_for_git(raw_patch, working_dir):
    normalized_repo_path = normalize_repo_path(working_dir)
    patch = replace_repo_paths_in_patch(raw_patch, normalized_repo_path)
    patch = fix_syntax_errors_in_patch(patch)
    patch = filter_and_reconstruct_patch_hunks(patch)
    patch = append_spaces_to_lines(patch)
    patch = strip_whitespace_from_change_lines(patch)
    patch = replace_line_numbers_in_hunk_headers(patch)

    # Final adjustments
    patch = patch.strip("\n") + "\n"

    return patch

def colored_diff(expected, actual):
    diff = difflib.unified_diff(expected.splitlines(), actual.splitlines(), lineterm='\n', fromfile='Expected', tofile='Actual')
    return ''.join(
        '\033[91;1m' + line + '\n' if line.startswith('-') else
        '\033[92;1m' + line + '\n' if line.startswith('+') else
        '\033[0;0m' + line + '\n'
        for line in diff)

def test_prepare_patch_for_git(input_patch, expected_output):
    result = prepare_patch_for_git(input_patch, "/fake/working/dir")
    if result != expected_output:
        logger.error(f"Test failed.")
        print(colored_diff(expected_output, result))
        return False
    return True

if __name__ == "__main__":
    result = test_prepare_patch_for_git(open("tests/1_input.patch", "r").read(), open("tests/1_expected.patch", "r").read())
    print(result)

    # input_2 = json.loads(open("tests/2_input.json", "r").read())
    # result = test_prepare_patch_for_git(input_2, )
    # print(result)
