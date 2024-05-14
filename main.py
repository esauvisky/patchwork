#!/usr/bin/env python3
from pdb import run
from InquirerPy.resolver import prompt
from InquirerPy import inquirer as inquirer
from InquirerPy.validator import PathValidator
import tempfile
import os
import tempfile
from subprocess import call
import subprocess
import io
import os
import re
from openai import OpenAI
import json
from unidiff import PatchSet
from git import GitCommandError, Repo, Git
import constants
from utils import convert_files_dict, extract_codeblocks, get_gitignore_files, get_user_prompt, select_files, select_options, select_user_files, validate_git_repo, run

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
    def __init__(self, system_message, name, model="gpt-4o", temperature=0.1):
        self.name = name
        self.system_message = system_message
        self.model = model
        self.temperature = temperature

    def send_messages(self, messages, max_attempts=3, temperature=None):
        response = client.chat.completions.create(model=self.model,
                                                  messages=[{"role": "system", "content": self.system_message}] + messages,
                                                  temperature=temperature if temperature else self.temperature,
                                                  max_tokens=4096,
                                                  response_format={"type": "json_object"},
                                                  stream=True)
        formatted_message = str(messages).replace('\\n', "\n")
        logger.trace(f"Agent {self.name} sending messages: {formatted_message}")
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

        output = json.loads("".join(output))
        return output


class Coordinator:
    def __init__(self, coordinator_agent: Agent, agents: List[Agent], directory_path: str):
        self.coordinator_agent = coordinator_agent
        self.suggestor_agent = agents[0]
        self.editor_agent = agents[1]
        self.repo = Repo(directory_path)

    def run(self, user_prompt, filepaths):
        project_files = self.get_file_contents(filepaths)
        user_input = json.dumps({"files": project_files, "prompt": user_prompt})
        coordinator_output = self.coordinator_agent.send_messages([{"role": "user", "content": user_input}])

        goal_files = self.get_file_contents(coordinator_output['filepaths'])
        suggestor_input = json.dumps({"files": goal_files, "goal": coordinator_output['goal']})
        suggestor_output = self.suggestor_agent.send_messages([{"role": "user", "content": suggestor_input}])

        tasks = suggestor_output['tasks']
        successful_patches = []
        for task in tasks:
            patches = self.get_patches(task)
            patches_files = [self.prepare_patch_for_git(patch) for patch in patches]
            successful_patches += self.apply_patches(patches_files)
        self.finalize(successful_patches)

    def apply_patches(self, patches):
        successful_patches = []
        for patch in patches:
            if not self.apply_patch(patch):
                action = self.handle_patch_failure(patch)
                if action == "Retry":
                    successful_patches += self.apply_patches([patch]) # Recursive retry
                elif action == "Edit":                                # Allow user to modify the patch or task details
                    edited_patch = self.edit_patch(patch)
                    successful_patches += self.apply_patches([edited_patch])
                elif action == "Skip":
                    continue
            else:
                successful_patches.append(patch)
        return successful_patches

    def handle_patch_failure(self, patch):
        choices = ["Retry", "Edit", "Skip"]
        questions = [{
            'type': 'list', 'name': 'response', 'message': 'Patch application failed. Choose an action:', 'choices': choices}]
        response = prompt(questions)
        return response['response']

    def edit_patch(self, patch):
        # Create a temporary file to hold the patch
        with tempfile.NamedTemporaryFile(delete=False, suffix=".patch", mode='w+') as tf:
            tf.write(patch)
            temp_file_path = tf.name

        # Open the temporary file in the default editor
        editor = os.environ.get('EDITOR', 'subl')
        call([editor, temp_file_path])

        # After editing, read the modified patch back from the file
        with open(temp_file_path, 'r') as tf:
            edited_patch = tf.read()

        # Optionally, clean up the temporary file
        os.remove(temp_file_path)

        return edited_patch

    def get_file_contents(self, file_paths):
        files_dict = {}

        for file_path in file_paths:
            try:
                absolute_path = ""
                if os.path.exists(os.path.join(self.repo.working_dir, file_path)):
                    # Check if the file exists at the root of the git repository
                    absolute_path = os.path.abspath(os.path.join(self.repo.working_dir, file_path))
                elif os.path.exists(file_path):
                    # Check if the file exists in the current directory
                    absolute_path = os.path.abspath(file_path)

                if not os.path.exists(absolute_path):
                    logger.warning(f"Error: File {absolute_path} does not exist. Will create a new empty file.")
                    with open(absolute_path, 'w') as file:
                        file.write("")
                        files_dict[absolute_path] = ""
                    continue
                else:
                    with open(absolute_path, 'r') as file:
                        files_dict[absolute_path] = file.read()
                    logger.trace(f"Found file: {absolute_path}. File size: {os.path.getsize(absolute_path)} bytes.")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        logger.info(f"Selected {len(file_paths)} files. Total size: {sum([os.path.getsize(file) for file in file_paths])/1024:.2f} kB.")

        return files_dict

    def apply_patch(self, patch):
        try:
            cmd = f"git apply --recount --verbose --unidiff-zero -C0 --allow-overlap --reject --ignore-space-change --ignore-whitespace --whitespace=warn --inaccurate-eof {patch}"
            run(cmd)
            logger.success("Patch applied successfully.")
        except Exception as e:
            logger.error(f"Patch application failed: {str(e)}. Attempting to fix and reapply.")
            return False
        return True

    def handle_task_selection(self, tasks):
        tasks_descriptions = [task["prompt"] for task in tasks]
        selected_indices = inquirer.checkbox(message="Please select the refactoring tasks you want to proceed with:", # type: ignore
                                             choices=[{"name": desc, "value": idx}
                                                      for idx, desc in enumerate(tasks_descriptions)]).execute()
        return [tasks[i] for i in selected_indices]

    def finalize(self, successful_patches):
        if successful_patches:
            self.repo.git.commit('-am', 'Applied successful patches')
        self.repo.git.checkout() # Return to the main branch or clean up

    def get_patches(self, task, error=None, temperature=None):
        prompt = f'{task["prompt"]}\n{task["info"]}'
        files = self.get_file_contents(task["filepaths"])

        if error:
            prompt += f"\n\nThe error was: {error}. Please try again."

        message = json.dumps({"files": files, "prompt": prompt})
        editor_messages = [{"role": "user", "content": message}]
        editor_output = self.editor_agent.send_messages(editor_messages, temperature=temperature)
        patches = editor_output['patches']

        # further split patches at the diff hunk boundary
        all_patches = []
        for patch in patches:
            splits = patch.split("diff --git")
            if len(splits) > 1:
                for split in splits[1:]:
                    all_patches.append("diff --git" + split)
            else:
                all_patches.append(patch)

        patches = all_patches

        logger.info(f"{len(patches)} patches for this task")
        return patches

    def prepare_patch_for_git(self, raw_patch):
        patch = raw_patch.replace("\\n", "\n") + "\n"
        # Normalize the repository working directory path to an absolute path without a trailing slash
        normalized_repo_path = os.path.abspath(self.repo.working_dir).rstrip('/').lstrip('/')
        escaped_repo_base_path = re.escape(normalized_repo_path)

        # new_patch = patch.replace(normalized_repo_path, "")
        # do it with regex, replacing any occurrence of the normalized repo path with an empty string
        new_patch = re.sub(escaped_repo_base_path + "/", "", patch, flags=re.MULTILINE)

        # Check if there were changes to the patch
        if new_patch == patch:
            logger.info(f"Patch filepaths were not modified.")
        else:
            logger.warning(f"Patch filepaths were modified. Stripped the following from the patch file:\n{escaped_repo_base_path}")

        # Check if are there any changing +/- lines that have no content
        if re.search(r"^[+-]\s+$", new_patch, flags=re.MULTILINE):
            # strip whitespaces from these lines, keeping the + and - signs intact
            new_patch = re.sub(r"^(\s*[+-])\s+$", r"\1", new_patch, flags=re.MULTILINE)
            logger.warning(f"Patch had changes only in whitespace. Stripped them from the patch file.")

        # Check for hunk headers that contain line numbers and replace them with @@ ... @@ to avoid conflicts
        # e.g.: from "@@ -19,7 +18,6 @@" to "@@ -0,0 +0,0 @@"
        if re.search(r"^@@ [-+\d\,\s]+ @@", new_patch, flags=re.MULTILINE):
            # strip the line numbers from the hunk headers
            new_patch = re.sub(r"^@@ [-+\d\,\s]+ @@", r"@@ -0,0 +0,0 @@", new_patch, flags=re.MULTILINE)
            logger.warning(f"Patch had hunk headers with line numbers. Replaced them with @@ ... @@.")

        # Write the modified patch to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(new_patch)
            temp_file_name = temp_file.name

        return temp_file_name

    def get_fixed_patch(self, files_contents, patch, error):
        user_input = files_contents + "\n\n"
        message = f"The following patch failed to apply:\n```patch\n{patch}\n```\n\nThe error was: {error}. Please return a new patch."
        logger.info(message)
        user_input += message

        text, codeblocks = self.editor_agent.send_messages([{"role": "user", "content": user_input}])

        patch = [
            codeblock["content"]
            for codeblock in codeblocks
            if codeblock["language"] == "patch" or codeblock["language"] == "diff"][0]
        logger.info(f"Got a patch for fixing the previous patch:\n```patch\n{patch}\n```")
        return patch


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

    selected_files = select_user_files(file_paths) if len(sys.argv) == 2 else file_paths
    prompt = get_user_prompt()

    agent_coordinator = Agent(system_message=constants.SYSTEM_MESSAGES["agent_coordinator"],
                              name="agent_coordinator",
                              temperature=0)
    agent_suggestor = Agent(system_message=constants.SYSTEM_MESSAGES["agent_suggestor"],
                            name="agent_suggestor",
                            temperature=0)
    agent_editor = Agent(system_message=constants.SYSTEM_MESSAGES["agent_editor"], name="agent_editor", temperature=0)
    coordinator = Coordinator(agent_coordinator, agents=[agent_suggestor, agent_editor], directory_path=directory_path)

    coordinator.run(prompt, selected_files)

    print("The refactoring was successful. The source code files have been updated with the patches.")


if __name__ == "__main__":
    main()
