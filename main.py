#!/usr/bin/env python3
import os
from openai import OpenAI
import json
from git import GitCommandError, Repo
import constants
from utils import convert_files_dict, extract_codeblocks, get_file_contents, get_file_contents_and_copy, get_gitignore_files, get_user_prompt, select_files, validate_git_repo

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from typing import List

import sys


class Agent:
    def __init__(self, system_message, name, model="gpt-4-turbo-preview"):
        self.name = name
        self.system_message = system_message
        self.model = model

    def send_messages(self, messages, max_attempts=3):
        response = client.chat.completions.create(model=self.model,
                                                  messages=[{"role": "system", "content": self.system_message}] + messages,
                                                  temperature=0.5,
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
                    print(f"Error: Agent {self.name} did not provide a valid response after {max_attempts} attempts.")
                    break
        sys.stdout.write("\n")
        output = "".join(output)
        return output, extract_codeblocks(output)


class Coordinator:
    def __init__(self, coordinator_agent: Agent, agents: List[Agent], directory_path):
        self.coordinator_agent = coordinator_agent
        self.agents = agents
        self.repo = Repo(directory_path)
        self.branch = self.repo.create_head('refactoring')

    def apply_patch(self, patch):
        try:
            self.repo.git.apply(patch)
            self.repo.git.add(update=True)
            self.repo.index.commit("Apply patch", parent_commits=(self.branch.commit,))
            return True
        except GitCommandError as e:
            print(f"Error applying patch: {e}")
            self.repo.git.reset('--hard')
            return False

    def finalize(self):
        self.branch.checkout()
        self.repo.git.merge('refactoring')

    def run(self, file_paths, user_prompt):
        while True:
            file_contents, file_paths = get_file_contents(file_paths)
            user_input = file_contents + "\n\n" + user_prompt

            # Step 1: Coordinator Agent
            coordinator_message = {"role": "user", "content": user_input}
            text, codeblocks = self.coordinator_agent.send_messages([coordinator_message])

            try:
                response = json.loads(codeblocks[0]["content"])
                relevant_files, relevant_file_paths = get_file_contents(response["files"])
            except json.JSONDecodeError:
                print(f"Error: The coordinator agent did not return a valid JSON object: {response}")
                return


            # Step 2: Suggestor Agent
            suggestor_agent = self.agents[0]
            suggestor_message = {"role": "user", "content": json.dumps(relevant_files)}
            text, codeblocks = suggestor_agent.send_messages([suggestor_message])

            try:
                tasks = json.loads(codeblocks[0]["content"])["tasks"]
            except json.JSONDecodeError:
                print(f"Error: The suggestor agent did not return a valid JSON object: {text}")
                return

            editor_agent = self.agents[1]
            for task in tasks:
                editor_messages = []

                prompt = task["prompt"]
                files_contents, file_paths = get_file_contents(task["files"])
                user_input = files_contents + "\n\n" + prompt

                editor_messages.append({"role": "user", "content": user_input})
                text, codeblocks = editor_agent.send_messages(editor_messages)

                # Apply each patch individually
                for patch in codeblocks:
                    success = False
                    attempts = 0
                    while not success and attempts < 3:
                        try:
                            print(f"Applying patch: {patch['language']}: {patch['content']}")
                            self.apply_patch(patch["content"])
                            success = True
                        except Exception as e:
                            attempts += 1
                            if attempts == 3:
                                print(f"Failed to apply patch after 3 attempts: {e}")
                            else:
                                print(f"Error applying patch, attempt {attempts}. Asking the editor for a new patch.")
                                editor_messages.append({"role": "user", "content": f"Patch:\n{patch}\n\nError:{e}\nYour patch failed to apply. Please provide a new patch."})
                                text, codeblocks = editor_agent.send_messages(editor_messages)
                                for patch in codeblocks:
                                    self.apply_patch(patch["content"])

def main():
    directory_path = sys.argv[1]
    if not os.path.exists(directory_path):
        print(f"Error: {directory_path} does not exist.")
        sys.exit(1)
    validate_git_repo(directory_path)
    ignored_files = get_gitignore_files(directory_path)
    file_paths = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, file)) and os.path.join(directory_path, file) not in ignored_files]
    prompt = get_user_prompt()

    agent_coordinator = Agent(system_message=constants.SYSTEM_MESSAGES["agent_coordinator"], name="agent_coordinator")
    agent_suggestor = Agent(system_message=constants.SYSTEM_MESSAGES["agent_suggestor"], name="agent_suggestor")
    agent_editor = Agent(system_message=constants.SYSTEM_MESSAGES["agent_editor"], name="agent_editor")
    coordinator = Coordinator(agent_coordinator, agents=[agent_suggestor, agent_editor], directory_path=directory_path)

    coordinator.run(file_paths, prompt)

    print("The refactoring was successful. The source code files have been updated with the patches.")


if __name__ == "__main__":
    main()
