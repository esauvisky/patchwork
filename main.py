#!/usr/bin/env python3
from pdb import run
import pprint
import random
import time
from InquirerPy.resolver import prompt
from InquirerPy import inquirer as inquirer
import tempfile
import os
import tempfile
from subprocess import call
import os
from google.api_core.exceptions import ResourceExhausted
import re
from openai import OpenAI
import json
from git import Repo
import constants
import codecs
from patch import prepare_patch_for_git
from utils import generate_markdown, get_gitignore_files, get_user_prompt, select_user_files, validate_git_repo, run

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from typing import List

import sys
import threading

from loguru import logger
import tqdm.auto as tqdm
import sys

# Models and their respective token limits
MODEL_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4192,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4-1106-preview": 127514,
    "gpt-4o-mini": 127514,
    "gpt-4o": 127514,
    "gpt-4": 16384,
    "gemini-1.5-pro": 1048576,
    "gemini-1.5-pro-exp-0801": 2097152,
    "gemini-1.5-flash": 1048576,}


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


setup_logging("TRACE")


class Agent:
    # def __init__(self, name, model="gemini-1.5-pro", temperature=0.5):
    def __init__(self, name, model="gemini-1.5-pro-exp-0801", temperature=0.5):
        self.name = name
        self.system_message = constants.SYSTEM_MESSAGES[name]
        self.model = model
        self.message_history = []
        self.temperature = temperature
        self.gemini_config = {
            "temperature": temperature, "max_output_tokens": MODEL_TOKEN_LIMITS[model],
            "response_mime_type": "application/json", "response_schema": constants.AGENTS_SCHEMAS[name]}

    def send_messages(self, messages, max_attempts=3, temperature=None):
        logger.debug(f"Agent {self.name} messages:")
        print(messages[0][list(messages[0].keys())[-1]])

        if "gemini" in self.model:
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.gemini_config, # type: ignore
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,},
                system_instruction=self.system_message,
            )
            history = [{"role": m["role"], "parts": m["content"]} for m in messages]

            while True:
                try:
                    response = model.generate_content(history)
                except ResourceExhausted:
                    logger.error("Error: hit quota. Retrying in 5 seconds.")
                    time.sleep(5)
                    continue
                break
        else:
            response = client.chat.completions.create(model=self.model,
                                                    messages=[{
                                                        "role": "system", "content": self.system_message}] + messages,
                                                    temperature=temperature if temperature else self.temperature,
                                                    max_tokens=4096,
                                                    response_format={"type": "json_object"},
                                                    stream=True)

        output = []

        if "gemini" in self.model:
            output = [response.text] # type: ignore
            logger.debug(f"Gemini response: {output}")
        else:
            for chunk in response:
                token = "".join([part.text
                                for part in chunk.candidates[0].content.parts]) if "gemini" in self.model else chunk.choices[0].delta.content # type: ignore
                if token is not None:
                    output.append(token)
                    # print(token, end="", flush=True)
                    if chunk.choices[0].finish_reason == "length": # type: ignore
                        break
            # print("\r" + " " * len(output), end="\r")

        try:
            logger.info(f"Loading JSON response from agent {self.name}'s response")
            output = json.loads("".join(output))
        except Exception as e:
            logger.error(f"Error parsing JSON response from agent {self.name}: {e}")
            if "Unterminated string" in str(e):
                return self._handle_truncated_response(messages)
            else:
                randtemp = random.uniform(0.5, 1.0)
                logger.error(f"Retrying with a random temperature of {randtemp}")
                return self.send_messages(messages, max_attempts, temperature=randtemp)

        logger.debug(f"Agent {self.name} response:")
        print(output[list(output.keys())[-1]])
        return output

    def _handle_truncated_response(self, messages):
        logger.error(f"Error: Agent {self.name} ran out of tokens.")
        messages.append({"role": "user", "content": "Your response was truncated and too long. Return the TASK_TOO_BROAD error and split the task into smaller tasks. Do not include any codeblocks or any additional text."})
        if "gemini" in self.model:
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.gemini_config, # type: ignore
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,},
                system_instruction=self.system_message,
            )
            history = [{"role": m["role"], "parts": m["content"]} for m in messages]
            while True:
                try:
                    response = model.generate_content(history)
                except ResourceExhausted:
                    logger.error("Error: hit quota. Retrying in 5 seconds.")
                    time.sleep(5)
                    continue
                break
        else:
            response = client.chat.completions.create(model=self.model,
                                                    messages=[{
                                                        "role": "system", "content": self.system_message},] + messages,
                                                    temperature=0,
                                                    max_tokens=4096,
                                                    response_format={"type": "json_object"},
                                                    stream=True)
        new_output = []
        if "gemini" in self.model:
            new_output = [response.text] # type: ignore
            logger.debug(f"Gemini response: {new_output}")
        else:
            for chunk in response:
                token = "".join([part.text for part in chunk.candidates[0].content.parts]) if "gemini" in self.model else chunk.choices[0].delta.content # type: ignore
                if token is not None:
                    new_output.append(token)
                    print(token, end="", flush=True)
            print("\n")
        return json.loads("".join(new_output))

class Coordinator:
    def __init__(self, coordinator_agent: Agent, agents: List[Agent], directory_path: str):
        self.coordinator_agent = coordinator_agent
        self.suggestor_agent = agents[0]
        self.editor_agent = agents[1]
        self.checker_agent = agents[2]
        self.all_filepaths = []
        self.repo = Repo(directory_path)

    def run(self, user_prompt, filepaths):
        project_files = self.get_files_contents(filepaths)
        user_input = generate_markdown(project_files, user_prompt)
        coordinator_output = self.coordinator_agent.send_messages([{"role": "user", "content": user_input}])
        goal = coordinator_output['goal']
        filtered_filepaths = coordinator_output['filepaths']
        self.all_filepaths = [os.path.abspath(file) for file in filepaths] if not self.all_filepaths else self.all_filepaths

        tasks = self.get_tasks(goal, filtered_filepaths)
        self.process_tasks(tasks)

    def process_tasks(self, tasks):
        for task in tasks:
            try:
                patches = self.get_patches(task)
                self.apply_changes(task, patches)
            except Exception as e:
                if "TASK_TOO_BROAD" in str(e):
                    logger.warning(f"Task is too broad. Trying to narrow it down. Splitted into {len(e.args[0]["tasks"])} tasks.")
                    subtasks = []
                    for subtask in e.args[0]["tasks"]:
                        subtasks.append({"prompt": subtask['prompt'], "info": subtask['info'], "filepaths": task['filepaths']})
                    self.process_tasks(subtasks)
                else:
                    raise e

    def get_tasks(self, goal, filepaths):
        goal_files = self.get_files_contents(filepaths)
        suggestor_input = generate_markdown(goal_files, goal)
        suggestor_output = self.suggestor_agent.send_messages([{"role": "user", "content": suggestor_input}])
        return suggestor_output['tasks']

    def replace_file(self, file, content):
        try:
            file_path = os.path.join(self.repo.working_dir, file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            logger.success(f"Replaced content of file {file} successfully.")
        except Exception as e:
            logger.error(f"Error replacing file {file}: {e}")

    def apply_changes(self, task, patches, retry_count=2):
        if retry_count == 0:
            logger.critical("Tried 3 times to apply the patch. Giving up.")
            # sys.exit(1)

        if len(patches) > 0:
            while patches:
                patch = patches.pop(0)
                try:
                    self.apply_patch(patch)
                except Exception as e:
                    if "No such file or directory" in str(e):
                        pattern = r"error:\s(.*?):\sNo such file or directory"
                        filepath = re.search(pattern, str(e)).group(1) # type: ignore
                        logger.error(f"Error: File {filepath} doesn't exist. Creating it and trying again.")
                        with open(filepath, 'w') as f:
                            f.write("")
                        patches.insert(0, patch)
                        continue
                    logger.error(f"Error when applying git patch: ${e}. Trying again...")
                    fixed_patches = self.get_patches(task, error=e)
                    patches.insert(0, fixed_patches[0])
                    # thread = threading.Thread(target=handle_patch_failure, daemon=True)
                    # thread.start()
                    # thread.join(5)
                    # if action == "Retry" or action is None:
                    # if action == "Edit":
                    #     edited_patch = self.edit_patch(patch)
                    #     patches.insert(0, edited_patch) # Insert the edited patch at the beginning of the list
                    # elif action == "Skip":
                    #     continue
                else:
                    logger.success(f"Patch {patch} applied successfully.")

    def edit_patch(self, patch): # Open the temporary file in the default editor
        editors = ["subl", "subl3", "code", os.environ.get('EDITOR')]
        editor = [editor for editor in editors if os.path.exists(f"/usr/bin/{editor}" or f"/usr/local/bin/{editor}")][0]
        call([editor, "-w", patch])
        return patch

    def get_files_contents(self, file_paths):
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
                else:
                    if self.repo.working_dir in file_path:
                        absolute_path = os.path.abspath(file_path)
                    else:
                        absolute_path = os.path.join(self.repo.working_dir, file_path)

                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

                if not os.path.exists(absolute_path):
                    logger.warning(f"Error: File {absolute_path} does not exist.")
                    with open(absolute_path, 'w') as f:
                        f.write("")

                relative_path = os.path.relpath(absolute_path, self.repo.working_dir)
                with open(absolute_path, 'r') as file:
                    # files_dict[relative_path] = codecs.encode(file.read(), "unicode-escape").decode("utf-8")
                    files_dict[relative_path] = file.read()
                logger.trace(f"Found file: {relative_path}. File size: {os.path.getsize(absolute_path)} bytes.")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                raise e

        logger.info(f"Selected {len(file_paths)} files. Total size: {sum([os.path.getsize(file) for file in file_paths])/1024:.2f} kB.")
        return files_dict

    def apply_patch(self, patch):
        try:
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
                temp_file.write(patch.encode("utf-8"))
            cmd = f"git apply --recount --verbose -C1 {temp_file.name}"
            run(cmd)
            logger.success("Patch applied successfully.")
        except:
            logger.error("Error applying patch, trying to escape special characters.")
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
                patch = codecs.decode(patch, "unicode-escape")
                # # patch = patch.replace("\\", "\\\\")
                # patch = patch.replace("\\n", "\n")
                # patch = patch.replace("\\r", "\r")
                # patch = patch.replace('\\"', '\"')
                temp_file.write(patch.encode("utf-8"))
            cmd = f"git apply --recount --verbose -C1 {temp_file.name}"
            run(cmd)
            logger.success("Patch applied successfully.")

    def get_patches(self, task, error=None, temperature=None):
        logger.info(f"Task: {task['prompt']}")
        prompt = f'{task["prompt"]}\n{task["info"]}'
        files = self.get_files_contents(task["filepaths"])

        if error:
            prompt += f"\n\nThere was an error while applying this patch: {error}. Please create a new patch to retry the failed hunks. Break the patch into more hunks of smaller size, even if contexts overlap."

        message = generate_markdown(files, prompt)
        # Append the new message to the history
        editor_messages = [{"role": "user", "content": message}]
        editor_output = self.editor_agent.send_messages(editor_messages, temperature=temperature)

        if "error" in editor_output and editor_output["error"]:
            raise Exception(editor_output)
        elif "patch" not in editor_output or len(editor_output["patch"]) == 0:
            raise Exception("NO_PATCHES")
        else:
            patches_contents = [prepare_patch_for_git(editor_output['patch'], self.repo.working_dir)]
            return patches_contents

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process directory and file paths.")
    parser.add_argument("directory_path", type=str, help="The directory path.")
    parser.add_argument("files", nargs="*", help="Specific files to include.")
    args = parser.parse_args()

    directory_path = sys.argv[1]
    if not os.path.exists(directory_path):
        print(f"Error: {directory_path} does not exist.")
        sys.exit(1)

    validate_git_repo(directory_path)
    ignored_files = get_gitignore_files(directory_path)

    if len(sys.argv) > 2:
        # Specific files are provided as arguments
        file_paths = []
        for file in args.files:
            if os.path.isdir(file):
                file_paths.extend([os.path.join(file, f) for f in os.listdir(file)])
            elif os.path.isfile(os.path.join(directory_path, file)) and os.path.join(directory_path, file) not in ignored_files:
                file_paths.append(os.path.join(directory_path, file))
    else:
        # No specific files provided, use all files not in .gitignore
        file_paths = [
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, file)) and os.path.join(directory_path, file) not in ignored_files]

    selected_files = select_user_files(file_paths) if len(sys.argv) == 2 else file_paths
    prompt = get_user_prompt()

    agent_coordinator = Agent(name="agent_coordinator", model="gpt-4o-mini", temperature=1)
    agent_suggestor = Agent(name="agent_suggestor", model="gpt-4o-mini", temperature=1)
    agent_editor = Agent(name="agent_editor", model="gpt-4o-mini", temperature=1)
    agent_checker = Agent(name="agent_checker", model="gpt-4o-mini", temperature=1)

    coordinator = Coordinator(agent_coordinator,
                              agents=[agent_suggestor, agent_editor, agent_checker],
                              directory_path=directory_path)
    coordinator.run(prompt, selected_files)

    print("The refactoring was successful. The source code files have been updated with the patches.")


if __name__ == "__main__":
    main()
