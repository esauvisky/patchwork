import sys
import re
import os
from git import Repo, InvalidGitRepositoryError
import shutil
import pathspec

file_cache = {}


def validate_git_repo(path):
    try:
        Repo(path)
    except InvalidGitRepositoryError:
        print(f"Error: {path} is not a valid Git repository.")
        sys.exit(1)


def select_files(directory_path):
    files = os.listdir(directory_path)
    print("Select the files you want to refactor:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    selected_files = input("Enter the numbers of the files you want to refactor, separated by commas: ")
    selected_files = selected_files.split(",")
    file_paths = [os.path.join(directory_path, files[int(i) - 1]) for i in selected_files]
    return file_paths


def get_user_prompt():
    print("Enter a prompt to use for the refactoring. The prompt should describe the refactoring task you want to perform.")
    prompt = input("Prompt: ")
    return prompt


def get_file_contents_and_copy(file_paths):
    temp_dir = "./files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_contents = ""
    file_paths_new = []
    for file_path in file_paths:
        try:
            # Convert the file path to an absolute path
            file_path = os.path.abspath(file_path)
            # Copy each file to the temporary directory
            temp_file_path = shutil.copy(file_path, temp_dir)
            with open(temp_file_path, 'r') as file:
                file_paths_new.append(temp_file_path)
                file_contents += f"Path: {temp_file_path}\n```{os.path.basename(temp_file_path).split('.')[1]}\n{file.read()}\n```\n"
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return file_contents, file_paths_new


def get_gitignore_files(directory_path):
    with open(os.path.join(directory_path, '.gitignore'), 'r') as f:
        gitignore = f.read()
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, gitignore.splitlines())
    all_files = [os.path.join(root, name) for root, dirs, files in os.walk(directory_path) for name in files]
    ignored_files = list(spec.match_files(all_files))
    return ignored_files


def convert_files_dict(file_dict):
    result = {"files": {}}
    for file_path in file_dict["files"]:
        with open(file_path, "r") as file:
            result["files"][file_path] = file.read()
    return result


def extract_codeblocks(text):
    # Find all codeblocks, capture both the content and the language
    codeblocks = re.findall(r'```(?P<language>\w+)\n(?P<content>.*?)```', text, re.DOTALL)

    # Return a list of codeblocks objects, each with the content and language
    return [{"content": codeblock, "language": language} for language, codeblock in codeblocks]
