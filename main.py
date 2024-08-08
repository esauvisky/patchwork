#!/usr/bin/env python3
from pdb import run
import pprint
import random
from InquirerPy.resolver import prompt
from InquirerPy import inquirer as inquirer
import tempfile
import os
import tempfile
from subprocess import call
import os
import re
from openai import OpenAI
import json
from git import Repo
import constants
import codecs
from utils import get_gitignore_files, get_user_prompt, select_user_files, validate_git_repo, run

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
    "gpt-4o": 127514,
    "gpt-4": 16384,
    "gemini-1.5-pro": 1048576,
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
    def __init__(self, name, model="gpt-4o", temperature=0.5):
        self.name = name
        self.system_message = constants.SYSTEM_MESSAGES[name]
        self.model = model
        self.temperature = temperature
        self.gemini_config = {
            "temperature": temperature, "max_output_tokens": MODEL_TOKEN_LIMITS[model],
            "response_mime_type": "application/json", "response_schema": constants.AGENTS_SCHEMAS[name]}

    def send_messages(self, messages, max_attempts=3, temperature=None):
        if "gemini" in self.model:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config=self.gemini_config,                                                                                                      # type: ignore
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,},
                system_instruction=self.system_message + "\n\nSerialize the JSON response to a string, escape newlines, tabs, and other special characters, and return it.",
            )
            history = [{"role": m["role"], "parts": m["content"]} for m in messages]
            response = model.generate_content(history, stream=True)                                                                                        # type: ignore
        else:
            response = client.chat.completions.create(model=self.model,
                                                      messages=[{
                                                          "role": "system", "content": self.system_message + "\n\nSerialize the JSON response to a string, escape newlines, tabs, and other special characters, and return it."}] + messages,
                                                      temperature=temperature if temperature else self.temperature,
                                                      max_tokens=4096,
                                                      response_format={"type": "json_object"},
                                                      stream=True)

        output = []
        for chunk in response:
            token = "".join([part.text
                             for part in chunk.candidates[0].content.parts]).replace("\n", "\\n") if "gemini" in self.model else chunk.choices[0].delta.content # type: ignore
            if token is not None:
                output.append(token)
                print(token, end="", flush=True)
                if chunk.choices[0].finish_reason == "length": # type: ignore
                    break
        print("\n")

        try:
            logger.info(f"Loading JSON response from agent {self.name}'s response")
            output = json.loads("".join(output))
        except Exception as e:
            logger.error(f"Error parsing JSON response from agent {self.name}: {e}")
            if "Unterminated string" in str(e):
                logger.error(f"Error: Agent {self.name} ran out of tokens.")
                messages.append({"role": "assistant", "content": "".join(output)})
                messages.append({"role": "user", "content": "Continue the previous JSON response starting exactly at the character where you left off, without codeblocks or any additional text."})
                response = client.chat.completions.create(model=self.model,
                                                        messages=[{
                                                            "role": "system", "content": self.system_message},] + messages,
                                                        temperature=0,
                                                        max_tokens=4096,
                                                        #   response_format={"type": "json_object"},
                                                        stream=True)
                for chunk in response:
                    token = chunk.choices[0].delta.content # type: ignore
                    if token is not None:
                        output.append(token)
                        print(token, end="", flush=True)
                print("\n")
                output = json.loads("".join(output))
            else:
                randtemp = random.uniform(0.5, 1.0)
                logger.error(f"Retrying with a random temperature of {randtemp}")
                return self.send_messages(messages, max_attempts, temperature=randtemp)

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

    def run(self, user_prompt, filepaths, retry_count=3):
        project_files = self.get_files_contents(filepaths)
        user_input = json.dumps({"files": project_files, "prompt": user_prompt})
        coordinator_output = self.coordinator_agent.send_messages([{"role": "user", "content": user_input}])
        goal = coordinator_output['goal']
        filtered_filepaths = coordinator_output['filepaths']
        self.all_filepaths = [os.path.abspath(file) for file in filepaths] if not self.all_filepaths else self.all_filepaths

        if retry_count == 0:
            logger.critical("Tried 3 times to make tasks less broad. Giving up.")
            sys.exit(1)

        tasks = self.get_tasks(goal, filtered_filepaths)
        for task in tasks:
            try:
                patches_filepaths = self.get_patches_filepaths(task)
                self.apply_changes(task, patches_filepaths)
            except Exception as e:
                if "TASK_TOO_BROAD" in str(e):
                    logger.warning(f"Task {task['prompt']} is too broad. Trying to narrow it down.")
                    task_files = self.get_files_contents(task['filepaths'])
                    task_goal = task['prompt'] + "\n\n" + task['info']
                    self.run(task_goal, task_files, retry_count-1)
                else:
                    raise e

    def get_tasks(self, goal, filepaths):
        goal_files = self.get_files_contents(filepaths)
        suggestor_input = json.dumps({"files": goal_files, "goal": goal})
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
            sys.exit(1)

        if len(patches) > 0:
            while patches:
                patch = patches.pop(0)
                try:
                    self.apply_patch(patch)
                except Exception as e:
                    logger.error(f"Error when applying git patch: ${e}. Trying again...")
                    fixed_patches = self.get_patches_filepaths(task, error=e)
                    self.apply_changes(task, fixed_patches, retry_count-1)
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
                    logger.warning(f"Error: File {file_path} does not exist.")
                    if self.repo.working_dir in file_path:
                        absolute_path = os.path.abspath(file_path)
                    else:
                        absolute_path = os.path.join(self.repo.working_dir, file_path)

                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

                if not os.path.exists(absolute_path):
                    logger.warning(f"Error: File {absolute_path} does not exist.")
                    with open(absolute_path, 'w') as f: f.write("")

                relative_path = os.path.relpath(absolute_path, self.repo.working_dir)
                with open(absolute_path, 'r') as file:
                    files_dict[relative_path] = file.read()
                logger.trace(f"Found file: {relative_path}. File size: {os.path.getsize(absolute_path)} bytes.")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                raise e

        logger.info(f"Selected {len(file_paths)} files. Total size: {sum([os.path.getsize(file) for file in file_paths])/1024:.2f} kB.")

        return files_dict

    def apply_patch(self, patch):
        cmd = f"git apply --recount --verbose -C0 --reject {patch}" # --inaccurate-eof --unidiff-zero --ignore-space-change --ignore-whitespace --whitespace=fix
        run(cmd)
        logger.success("Patch applied successfully.")

    def get_patches_filepaths(self, task, error=None, temperature=None):
        logger.info(f"Getting patches for task {task['prompt']}")
        prompt = f'{task["prompt"]}\n{task["info"]}'
        files = self.get_files_contents(task["filepaths"])

        if error:
            prompt += f"\n\nThere was an error while applying the this patch: {error}. Please create a new patch to retry the failed hunks. Break the patch into more hunks of smaller size, even if contexts overlap."

        message = json.dumps({"files": files, "task": prompt})
        editor_messages = [{"role": "user", "content": message}]
        editor_output = self.editor_agent.send_messages(editor_messages, temperature=temperature)

        if "error" in editor_output:
            raise Exception(editor_output["error"])
        elif "patches" not in editor_output or len(editor_output["patches"]) == 0:
            raise Exception("NO_PATCHES")
        else:
            patches_paths = [self.prepare_patch_for_git(patch) for patch in editor_output['patches']]
            return patches_paths

    def prepare_patch_for_git(self, raw_patch):
        # patch = raw_patch.replace("\\n", "\n") + "\n
        patch = raw_patch
        # Normalize the repository working directory path to an absolute path without a trailing slash
        normalized_repo_path = os.path.abspath(self.repo.working_dir).rstrip('/').lstrip('/')
        escaped_repo_base_path = re.escape(normalized_repo_path)

        # new_patch = patch.replace(normalized_repo_path, "")
        # do it with regex, replacing any occurrence of the normalized repo path with an empty string
        new_patch = re.sub(escaped_repo_base_path + "/", "", patch, flags=re.MULTILINE)

        # Handle complex cases where paths might still contain parts of the absolute path
        normalized_repo_path_segments = normalized_repo_path.split('/')
        for i in range(len(normalized_repo_path_segments), 0, -1):
            partial_path = "/".join(normalized_repo_path_segments[:i])
            escaped_partial_path = re.escape(partial_path)
            new_patch = re.sub(rf"{escaped_partial_path}/", "", new_patch, flags=re.MULTILINE)

        # Check if there were changes to the patch
        if new_patch == patch:
            logger.info(f"Patch filepaths were not modified.")
        else:
            logger.warning(f"Patch filepaths were modified. Stripped the following from the patch file:\n{escaped_repo_base_path}")

        # Check if there are syntax errors in the patch (like no space before context lines or spaces before diff special characters)
        # Step 1: Remove leading space if the line matches the criteria
        match = re.search(r"^ (?=(?:[-+@]|diff --git|index ))", new_patch, flags=re.MULTILINE)
        if match:
            new_patch = re.sub(r"^ (?=([-+@]|diff --git|index ))", r"\1", new_patch, flags=re.MULTILINE)
            logger.warning(f"Patch had special lines with leading spaces. Stripped them from the patch file.")
            logger.debug(f"Match: {" ".join(match.groups())}")

        # Step 2: Append a space if the line does not match the criteria
        match = re.search(r"^(?!(?:[-+@ ]|diff --git|index |\n))(.*)$", new_patch, flags=re.MULTILINE)
        if match and not re.search(r"^\s*$", match.group(1)):
            logger.warning(f"Patch has context lines without leading space. Added a space to them")
            logger.debug(f"Match: {" ".join(match.groups())}")
            new_patch = re.sub(r"^(?!(?:[-+@ ]|diff --git|index |\n))(.*)$", r" \1", new_patch, flags=re.MULTILINE)

        # Check if are there any changing +/- lines that have no content
        # if re.search(r"^[+-]\s+$", new_patch, flags=re.MULTILINE):
        #     # strip whitespaces from these lines, keeping the + and - signs intact
        #     new_patch = re.sub(r"^(\s*[+-])\s+$", r"\1", new_patch, flags=re.MULTILINE)
        #     logger.warning(f"Patch had changes only in whitespace. Stripped them from the patch file.")

        # Check for hunk headers that contain line numbers and replace them with @@ ... @@ to avoid conflicts
        # e.g.: from "@@ -19,7 +18,6 @@" to "@@ -0,0 +0,0 @@"
        # if re.search(r"^@@ [-+\d\,\s]+ @@", new_patch, flags=re.MULTILINE):
        #     # strip the line numbers from the hunk headers
        #     new_patch = re.sub(r"^@@ [-+\d\,\s]+ @@", r"@@ -0,0 +0,0 @@", new_patch, flags=re.MULTILINE)
        #     logger.warning(f"Patch had hunk headers with line numbers. Replaced them with @@ -0,0 +0,0 @@.")

        # replace any unicode \uXXXX sequences with their corresponding characters
        new_patch = codecs.decode(new_patch, 'unicode_escape')

        # replace any sequence of \n at the end of the patch with a single \n
        # fixes the depends on old contents for new files
        new_patch = new_patch.strip("\n") + "\n"

        # Write the modified patch to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(new_patch)
            temp_file_name = temp_file.name

        return temp_file_name

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

    agent_coordinator = Agent(name="agent_coordinator", model="gpt-4o", temperature=1)
    agent_suggestor = Agent(name="agent_suggestor", model="gpt-4o", temperature=0.5)
    agent_editor = Agent(name="agent_editor", model="gpt-4o", temperature=0)
    agent_checker = Agent(name="agent_checker", model="gpt-4o", temperature=0)

    coordinator = Coordinator(agent_coordinator,
                              agents=[agent_suggestor, agent_editor, agent_checker],
                              directory_path=directory_path)
    coordinator.run(prompt, selected_files)

    print("The refactoring was successful. The source code files have been updated with the patches.")


if __name__ == "__main__":
    main()
