#!/usr/bin/env python3
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import tempfile
import subprocess
import io
import os
import re
from openai import OpenAI
import json
from unidiff import PatchSet
from git import GitCommandError, Repo
import constants
from utils import convert_files_dict, extract_codeblocks, get_gitignore_files, get_user_prompt, select_files, select_options, select_user_files, validate_git_repo

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from typing import List

import sys

from loguru import logger
import tqdm.auto as tqdm
import sys


def setup_logging(level="DEBUG", show_module=False):
    """
    Setups better log format for loguru
    """
    logger.remove(0)                                 # Remove the default logger
    log_level = level
    log_fmt = u"<green>["
    log_fmt += u"{file:10.10}…:{line:<3} | " if show_module else ""
    log_fmt += u"{time:HH:mm:ss.SSS}]</green> <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(lambda x: tqdm.tqdm.write(x, end=""),
               level=log_level,
               format=log_fmt,
               colorize=True,
               backtrace=True,
               diagnose=True)


setup_logging("DEBUG")


class Agent:
    def __init__(self, system_message, name, model="gpt-4-turbo-preview"):
        self.name = name
        self.system_message = system_message
        self.model = model

    def send_messages(self, messages, max_attempts=3):
        response = client.chat.completions.create(model=self.model,
                                                  messages=[{"role": "system", "content": self.system_message}] + messages,
                                                  temperature=1,
                                                  max_tokens=4096,
                                                  stream=True)

        output = []
        attempts = 0
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is not None:
                output.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()
            else:
                attempts += 1
                if attempts >= max_attempts:
                    logger.error(f"Error: Agent {self.name} did not provide a valid response after {max_attempts} attempts.")
                    break
        sys.stdout.write("\n")
        output = "".join(output)
        return output, extract_codeblocks(output)


class Coordinator:
    def __init__(self, coordinator_agent: Agent, agents: List[Agent], directory_path: str):
        self.coordinator_agent = coordinator_agent
        self.suggestor_agent = agents[0]
        self.editor_agent = agents[1]
        self.repo = Repo(directory_path)
        self.branch = self.repo.create_head('refactoring')

    def get_file_contents(self, file_paths):
        file_contents = ""
        for file_path in file_paths:
            try:
                if os.path.exists(os.path.join(self.repo.working_dir, file_path)):
                    # Check if the file exists at the root of the git repository
                    absolute_path = os.path.abspath(os.path.join(self.repo.working_dir, file_path))
                elif os.path.exists(file_path):
                    # Check if the file exists in the current directory
                    absolute_path = os.path.abspath(file_path)
                else:
                    # Create a new file at the root of the git repository
                    absolute_path = os.path.abspath(os.path.join(self.repo.working_dir, file_path))
                    # with open(absolute_path, 'w') as file:
                    #     file.write("")
                with open(absolute_path, 'r') as file:
                    file_contents += f"Path: {absolute_path}\n```{os.path.basename(absolute_path).split('.')[0] if len(os.path.basename(absolute_path).split('.')) == 1 else os.path.basename(absolute_path).split('.')[1]}\n{file.read()}\n```\n"
                logger.info(f"Found file: {absolute_path}. File size: {os.path.getsize(absolute_path)} bytes.")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        return file_contents, file_paths

    # def apply_patch(self, patch):
    #     # replace line numbers with '@@ -,0 @@'
    #     patch = re.sub(r'\n@@ .+ @@', '\n@@ -,0 @@', patch, re.MULTILINE)
    #     patch_set = PatchSet.from_string(patch)

    #     for patched_file in patch_set:
    #         file_path = "/" + patched_file.path
    #         try:
    #             # create if it doesn't exist
    #             if not os.path.exists(file_path):
    #                 with open(file_path, 'w') as file:
    #                     file.write("")
    #             with open(file_path, 'r+') as file:
    #                 lines = file.readlines()
    #                 for hunk in patched_file:
    #                     for line in hunk:
    #                         if line.is_added:
    #                             lines.insert(line.target_line_no - 1, line.value)
    #                         elif line.is_removed:
    #                             del lines[line.source_line_no - 1]
    #                 file.seek(0)
    #                 file.writelines(lines)
    #         except Exception as e:
    #             logger.error(f"Error applying patch to file {file_path}: {e}")
    #             return False

    #     return True

    def apply_patch(self, patch):
        try:
            return True
        except GitCommandError as e:
            logger.error(f"Error applying patch: {e}")
            self.repo.git.reset('--hard')
            return False
        finally:
            os.remove(temp_file_name)

    def finalize(self):
        self.branch.checkout()
        self.repo.git.merge('refactoring')

    def run(self, file_paths, user_prompt):
        file_contents, file_paths = self.get_file_contents(file_paths)
        user_input = file_contents + "\n\n" + user_prompt

        # Step 1: Coordinator Agent
        coordinator_message = {"role": "user", "content": user_input}
        text, codeblocks = self.coordinator_agent.send_messages([coordinator_message])

        try:
            response = json.loads(codeblocks[0]["content"])
            relevant_files, relevant_file_paths = self.get_file_contents(response["files"])
        except json.JSONDecodeError:
            logger.error(f"Error: The coordinator agent did not return a valid JSON object: {response}")
            return

        # Step 2: Suggestor Agent
        suggestor_message_directive = {"role": "user", "content": f"Use the following prompt as a directive for guidance on the intended goal of the refactoring process:\n\n{user_prompt}\n\n"}
        suggestor_message_contents = {"role": "user", "content": json.dumps(relevant_files)}
        text, codeblocks = self.suggestor_agent.send_messages([suggestor_message_directive, suggestor_message_contents])
        # After receiving tasks from the suggestor agent
        try:
            tasks = json.loads(codeblocks[0]["content"])["tasks"]
            tasks_descriptions = [task["prompt"] for task in tasks]  # Assuming each task has a 'description' field
        except json.JSONDecodeError:
            logger.error(f"Error: The suggestor agent did not return a valid JSON object: {text}")
            return

        # Present tasks to the user for selection
        print("Please select the refactoring tasks you want to proceed with:")
        selected_tasks_indices = select_options(tasks_descriptions)

        # Filter tasks based on user selection
        selected_tasks = [tasks[i] for i in selected_tasks_indices]

        # Proceed with only the selected tasks
        for task in selected_tasks:
            editor_messages = []

            prompt = task["prompt"]
            files_contents, file_paths = self.get_file_contents(task["files"])
            user_input = files_contents + "\n\n" + prompt

            editor_messages.append({"role": "user", "content": user_input})
            text, codeblocks = self.editor_agent.send_messages(editor_messages)

            # Apply each patch individually
            patches = [
                codeblock["content"]
                for codeblock in codeblocks
                if codeblock["language"] == "patch" or codeblock["language"] == "diff"]
            logger.info(f"Applying {len(patches)} patches")

            for ix, patch in enumerate(patches):
                success = False
                while not success:
                    try:
                        logger.info(f"Applying patch #{ix}")
                        # extract first line into message
                        message = patch.split("\n")[0]
                        # remove first line
                        patch = "\n".join(patch.split("\n")[1:])
                        # replace working_dir
                        patch = patch.replace(self.repo.working_dir, "")
                        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                            temp_file.write(patch)
                            temp_file_name = temp_file.name
                        self.repo.git.apply(temp_file_name,
                                            recount=True,
                                            allow_overlap=True,
                                            ignore_space=True,
                                            inaccurate_eof=True)
                        self.repo.git.add(update=True)
                        self.repo.index.commit(message, parent_commits=(self.branch.commit,))
                    except Exception as e:
                        logger.info(f"Failed to apply patch #{ix}:\n{e}")
                        logger.warning("What do you want to do with the patch file?")
                        inquirer.select("action", choices=["Edit", "Retry", "Skip"])
                        action = inquirer.get("action")
                        if action == "Edit":
                            # open with xdg-open and wait for user input
                            subprocess.run(["xdg-open", temp_file_name])
                            with open(temp_file_name, "r") as f:
                                patch = f.read()
                            # wait for user input
                            inquirer.wait_for_enter()
                            continue
                        elif action == "Retry":
                            with open(temp_file_name, "r") as f:
                                patch = f.read()
                            editor_messages.append({
                                "role": "user", "content": f"Patch:\n{patch}\n\nError:{e}\nYour patch failed to apply. Please provide a new patch."})
                            text, codeblocks = self.editor_agent.send_messages(editor_messages)
                            patch = next(codeblock["content"]
                                         for codeblock in codeblocks
                                         if codeblock["language"] == "patch" or codeblock["language"] == "diff")
                            continue
                        elif action == "Skip":
                            success = True
                            continue


def main():
    if len(sys.argv) < 2:
        print("Usage: script.py <directory_path> [file1 file2 ...]")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.exists(directory_path):
        print(f"Error: {directory_path} does not exist.")
        sys.exit(1)

    validate_git_repo(directory_path)
    ignored_files = get_gitignore_files(directory_path)

    if len(sys.argv) > 2:
        # Specific files are provided as arguments
        file_paths = [
            os.path.join(directory_path, file)
            for file in sys.argv[2:]
            if os.path.isfile(os.path.join(directory_path, file))]
    else:
        # No specific files provided, use all files not in .gitignore
        file_paths = [
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, file)) and os.path.join(directory_path, file) not in ignored_files]

    if not file_paths:
        print("No valid files selected or provided.")
        sys.exit(1)

    selected_files = select_user_files(file_paths) if len(sys.argv) == 2 else file_paths
    prompt = get_user_prompt()

    agent_coordinator = Agent(system_message=constants.SYSTEM_MESSAGES["agent_coordinator"], name="agent_coordinator")
    agent_suggestor = Agent(system_message=constants.SYSTEM_MESSAGES["agent_suggestor"], name="agent_suggestor")
    agent_editor = Agent(system_message=constants.SYSTEM_MESSAGES["agent_editor"], name="agent_editor")
    coordinator = Coordinator(agent_coordinator, agents=[agent_suggestor, agent_editor], directory_path=directory_path)

    coordinator.run(selected_files, prompt)

    print("The refactoring was successful. The source code files have been updated with the patches.")


if __name__ == "__main__":
    main()
