#!/usr/bin/env python3
from pdb import run
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
    def __init__(self, system_message, name, model="gpt-4-turbo-preview", temperature=0.1):
        self.name = name
        self.system_message = system_message
        self.model = model
        self.temperature = temperature

    def send_messages(self, messages, max_attempts=3, temperature=None):
        response = client.chat.completions.create(model=self.model,
                                                  messages=[{"role": "system", "content": self.system_message}] + messages,
                                                  temperature=temperature if temperature else self.temperature,
                                                  max_tokens=4096,
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
        output = "".join(output)
        return output, extract_codeblocks(output)


class Coordinator:
    def __init__(self, coordinator_agent: Agent, agents: List[Agent], directory_path: str):
        self.coordinator_agent = coordinator_agent
        self.suggestor_agent = agents[0]
        self.editor_agent = agents[1]
        self.repo = Repo(directory_path)
        # self.branch = self.repo.create_head('refactoring')

    def get_file_contents(self, file_paths):
        file_contents = ""
        for file_path in file_paths:
            try:
                absolute_path = ""
                if os.path.exists(os.path.join(self.repo.working_dir, file_path)):
                    # Check if the file exists at the root of the git repository
                    absolute_path = os.path.abspath(os.path.join(self.repo.working_dir, file_path))
                elif os.path.exists(file_path):
                    # Check if the file exists in the current directory
                    absolute_path = os.path.abspath(file_path)
                # else:
                if not os.path.exists(absolute_path):
                    logger.warning(f"Error: File {absolute_path} does not exist. Will create a new empty file.")
                    with open(absolute_path, 'w') as file:
                        file.write("")
                        file_contents += f"\n{absolute_path}:\n```{os.path.basename(absolute_path).split('.')[0] if len(os.path.basename(absolute_path).split('.')) == 1 else os.path.basename(absolute_path).split('.')[1]}\n\n```\n"
                    continue
                else:
                    with open(absolute_path, 'r') as file:
                        file_contents += f"\n{absolute_path}:\n```{os.path.basename(absolute_path).split('.')[0] if len(os.path.basename(absolute_path).split('.')) == 1 else os.path.basename(absolute_path).split('.')[1]}\n{file.read()}\n```\n"
                    logger.trace(f"Found file: {absolute_path}. File size: {os.path.getsize(absolute_path)} bytes.")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        logger.info(f"Selected {len(file_paths)} files. Total size: {sum([os.path.getsize(file) for file in file_paths])/1024:.2f} kB.")
        return file_contents, file_paths

    def finalize(self):
        pass
        # self.branch.checkout()
        # self.repo.git.merge('refactoring')

    def run(self, user_prompt, file_paths):
        file_contents, file_paths = self.get_file_contents(file_paths)
        user_input = file_contents + "\n\n" + user_prompt

        # Step 1: Coordinator Agent
        coordinator_message = {"role": "user", "content": user_input}
        _, codeblocks = self.coordinator_agent.send_messages([coordinator_message])
        response = json.loads(codeblocks[0]["content"])
        relevant_files_contents, _ = self.get_file_contents(response["files"])
        # goal = response["goal"]

        # Step 2: Suggestor Agent
        suggestor_message_contents = {
            "role": "user", "content": relevant_files_contents + "\n\n---------------------\n" + response["goal"]}
        text, codeblocks = self.suggestor_agent.send_messages([suggestor_message_contents])
        # After receiving tasks from the suggestor agent
        tasks = json.loads(codeblocks[0]["content"])["tasks"]
        tasks_descriptions = [task["prompt"] for task in tasks] # Assuming each task has a 'description' field

        # Present tasks to the user for selection
        print("Please select the refactoring tasks you want to proceed with:")
        selected_tasks_indices = select_options(tasks_descriptions, all_selected=True)
        selected_tasks = [tasks[i] for i in selected_tasks_indices]

        # Proceed with only the selected tasks
        for task in selected_tasks:
            patches = self.get_patches(task)

            ix = 0
            while patches:
                ix += 1
                raw_patch = patches.pop(0)

                patch_file = self.prepare_patch_for_git(raw_patch)

                try:
                    logger.info(f"Applying patch #{ix}")
                    cmd = f"git apply --recount --verbose --unidiff-zero -C0 --allow-overlap --reject --ignore-space-change --ignore-whitespace --whitespace=warn --inaccurate-eof {patch_file}"
                    stdout, stderr = run(cmd)
                except Exception as git_error:
                    logger.warning(git_error)
                    # Restore the previous state from stash
                    self.repo.git.checkout(task["files"])
                    # clear the patches list
                    patches = self.get_patches(task, temperature=1)
                    logger.warning(f"Trying again with {len(patches)} patches.")
                    continue
                    # try:
                    #     patch = self.prepare_patch_for_patch(raw_patch)
                    #     self.patch_with_patch(patch)
                    # except Exception as patch_error:
                    #     logger.error(f"Failed to apply patch with patch:\n{patch_error}\nTrying again...")
                    #     patches.insert(0, self.get_fixed_patch(task["files"], patch, git_error))
                    # else:
                    #     logger.success(f"Patch #{ix} applied successfully.")
                else:
                    logger.success(f"Patch #{ix} applied successfully:\n{stdout}")

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
        # @@ -19,7 +18,6 @@
        # to
        # @@ ... @@
        if re.search(r"^@@ [-+\d\,\s]+ @@", new_patch, flags=re.MULTILINE):
            # strip the line numbers from the hunk headers
            new_patch = re.sub(r"^@@ [-+\d\,\s]+ @@", r"@@ -0,0 +0,0 @@", new_patch, flags=re.MULTILINE)
            logger.warning(f"Patch had hunk headers with line numbers. Replaced them with @@ ... @@.")

        # Write the modified patch to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(new_patch)
            temp_file_name = temp_file.name

        return temp_file_name

    def patch_with_patch(self, temp_file_name):
        import subprocess

        try:
            # Save the current state
            self.repo.git.stash('save', "Stash before manual patching")

            # Execute the patch command
            patch_command = ["patch", "-E", "--merge", "--verbose", "-l", "-F5", "-i", temp_file_name]
            subproc = subprocess.run(patch_command, check=True, cwd=self.repo.working_dir, capture_output=True)
            logger.info("Patch applied successfully.")

        except subprocess.CalledProcessError as e:
            # Log error and output
            logger.error(f"Patch failed to apply. Error: {e.stderr.decode()}")

            # Restore previous state from stash
            self.repo.git.stash('pop')
            raise Exception(f"Patch failed to apply, reverted to previous state. Error: {e.stderr.decode()}")

        except Exception as e:
            # Log unexpected error
            logger.error(f"Unexpected error: {e}")

            # Restore previous state from stash
            self.repo.git.stash('pop')
            raise

    # def deal_with_patch_error(self, task, error):
    #     logger.warning("What do you want to do with the patch file?")
    #     action = inquirer.select("action", choices=["Edit", "Retry", "Skip"]).execute()
    #     if action == "Edit":
    #         # open with xdg-open and wait for user input
    #         # temp_file=tempfile.NamedTemporaryFile(mode='w+', delete=False)
    #         # temp_file.write(raw_patch)
    #         # temp_file.flush()
    #         subprocess.run(["xdg-open", temp_file.name])
    #         input("Press enter to continue...")
    #         with open(temp_file_name, "r") as f:
    #             patch = f.read()
    #         continue
    #     elif action == "Retry":
    #         with open(temp_file_name, "r") as f:
    #             patch = f.read()
    #         editor_messages.append({
    #             "role": "user", "content": f"Patch:\n{patch}\n\nError:{e}\nYour patch failed to apply. Please provide a new patch."})
    #         text, codeblocks = self.editor_agent.send_messages(editor_messages)
    #         patch = next(codeblock["content"]
    #                         for codeblock in codeblocks
    #                         if codeblock["language"] == "patch" or codeblock["language"] == "diff")
    #         continue
    #     elif action == "Skip":
    #         success = True
    #         continue

    def get_patches(self, task, error=None, temperature=None):
        prompt = task["prompt"]
        files_contents, file_paths = self.get_file_contents(task["files"])
        user_input = files_contents + "\n\n" + prompt + "\n\n"
        if error:
            user_input += f"The error was: {error}. Please try again."

        editor_messages = [{"role": "user", "content": user_input}]
        text, codeblocks = self.editor_agent.send_messages(editor_messages, temperature=temperature)

        patches = [
            codeblock["content"]
            for codeblock in codeblocks
            if codeblock["language"] == "patch" or codeblock["language"] == "diff"]
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
                              temperature=0.5)
    agent_suggestor = Agent(system_message=constants.SYSTEM_MESSAGES["agent_suggestor"],
                            name="agent_suggestor",
                            temperature=0.5)
    agent_editor = Agent(system_message=constants.SYSTEM_MESSAGES["agent_editor"], name="agent_editor", temperature=0.5)
    coordinator = Coordinator(agent_coordinator, agents=[agent_suggestor, agent_editor], directory_path=directory_path)

    coordinator.run(prompt, selected_files)

    print("The refactoring was successful. The source code files have been updated with the patches.")


if __name__ == "__main__":
    main()
