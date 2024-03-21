#!/usr/bin/env python3
import os
from openai import OpenAI
import json
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from typing import List

import sys


def get_file_contents(file_paths):
    file_contents = ""
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                file_contents += f"File: `{file_path}`\n```\n{file.read()}\n```\n\n"
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return file_contents


class Agent:
    def __init__(self, system_message, name, model="gpt-4-turbo-preview"):
        self.name = name
        self.system_message = system_message
        self.model = model

    def send_message(self, messages, max_attempts=3):
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_message}] + messages,
            temperature=0,
            stream=True                                                                   # Add this line to enable streaming
        )

        output = ""
        attempts = 0
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is not None:
                output += token
                sys.stdout.write(token)
                sys.stdout.flush()
            else:
                attempts += 1
                if attempts >= max_attempts:
                    print(f"Error: Agent {self.name} did not provide a valid response after {max_attempts} attempts.")
                    break

        return output

class Coordinator:
    def __init__(self, coordinator_agent: Agent, agents: List[Agent]):
        self.coordinator_agent = coordinator_agent
        self.agents = agents

    def run(self, user_input):
        messages = [{"role": "system", "content": self.coordinator_agent.system_message}, {"role": "user", "content": user_input + "\nPlease, return ONLY a JSON object like this:"+
        """
        {
            "files": [
                "file1.py",
                "file2.py",
                ...
            ],
            "agent": "refactor_suggestor or refactor_editor",
            "prompt": "your prompt"
        }

        "your prompt" is the prompt you want to send to the agent "refactor_suggestor" or "refactor_editor". Ask the agents to return JSON objects and nothing else, just like this.
        """}]
        coordinator_output = self.coordinator_agent.send_message(messages, max_attempts=5)
        if coordinator_output:
            try:
                coordinator_response = json.loads(coordinator_output.split("```json")[1].split("```")[0])
            except json.JSONDecodeError:
                print(f"Error: The coordinator agent did not return a valid JSON object: {coordinator_output}")
                return

            # 2. Present the source code in language format to the `refactor_suggestor`, whom will reply with required refactoring task that needs to be performed.
            suggestor_agent = self.agents[0]
            suggestor_messages = messages + [{"role": "user", "content": coordinator_response["prompt"]}]
            suggestor_output = suggestor_agent.send_message(suggestor_messages)
            tasks = [suggestor_output]
            messages.append({"role": "assistant", "content": suggestor_output})

            while tasks:
                # 3. For each task, present the latest version of the code alongside the task to the `refactor_editor` assistant. The assistant will create .patch files to be applied on top of the input files.
                task = tasks.pop(0)
                editor_agent = self.agents[1]
                editor_messages = messages + [{"role": "user", "content": task}]
                editor_output = editor_agent.send_message(editor_messages)
                messages.append({"role": "assistant", "content": editor_output})

                # # 4. Ask `refactor_runner` to execute the required shell commands (using `sh` codeblocks) that will apply the patches on top of the source code file(s). `refactor_runner` might require several consecutive runs. If it reaches 10 or replies with CONTINUE, it means it's job is done and you can proceed to step 4. If it replies with TERMINATE, stop everything.
                # runner_messages = messages + [{"role": "user", "content": "Please apply the patch files created by the refactor_editor and update the source code files."}]
                # runner_output = runner_agent.send_message(runner_messages)
                # messages.append({"role": "assistant", "content": runner_output})

                # if runner_output == "TERMINATE":
                #     break
                # elif runner_output != "CONTINUE":
                    # 5. Repeat steps 2-4 until `refactor_suggestor` doesn't have any more tasks
                suggestor_messages = messages + [{"role": "user", "content": "Please suggest the next refactoring task for the updated code."}]
                suggestor_output = suggestor_agent.send_message(suggestor_messages)
                if suggestor_output:
                    tasks.append(suggestor_output)
                    messages.append({"role": "assistant", "content": suggestor_output})

        return "\n".join(m["content"] for m in messages if m["role"] == "assistant")


if __name__ == "__main__":
    # Check if at least one file path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <file1> <file2> ... <fileN>")
        sys.exit(1)

    # Get the list of file paths from the command line arguments
    file_paths = sys.argv[1:]

    refactor_coordinator = Agent(system_message="As the coordinator, your role is to manage the workflow between the agents `refactor_suggestor` and `refactor_editor`. Ensure the code provided by the user is refactored according to best practices. Coordinate the agents to perform their tasks sequentially and track their progress.\n\nWorkflow steps:\n1. If the user provides file paths, read and present the code to `refactor_suggestor`.\n2. `refactor_suggestor` will analyze the code and suggest a specific refactoring task.\n3. Pass the task and code to `refactor_editor` to create patch files for the suggested refactoring.\n4. Apply the patches to the source code files.\n5. Repeat steps 2-4 until no further refactoring tasks are suggested.",
                            name="refactor_coordinator")
    refactor_suggestor = Agent(system_message="Your role is to identify and suggest specific refactoring tasks for the provided code. Focus on structural improvements that enhance readability, adhere to best practices, and optimize performance. Provide one task at a time, targeting significant changes rather than minor edits. Avoid abstract instructions; be precise and actionable.\n\nExample:\nInput: [code snippet]\nOutput: - Refactor `processData` to use a generator for memory efficiency.\n\nProvide only the task in your response for clear parsing. Do not attempt to run code or work with hypothetical scenarios. Prioritize tasks with the most impact, such as creating new files or restructuring large code blocks.\n",
                               name="refactor_suggestor")
    refactor_editor = Agent(system_message="Your responsibility is to create patch files based on the refactoring tasks provided. Ensure the patches are concise and apply cleanly to the input code. Use shell commands to save the patch files. Do not summarize or omit important code sections; split them into separate patches if necessary. If input files are missing, create them first. Focus on accuracy and avoid hypothetical code scenarios.\n",
                            name="refactor_editor")

    coordinator = Coordinator(refactor_coordinator, agents=[refactor_suggestor, refactor_editor])

    file_contents = get_file_contents(file_paths)
    prompt = input("Enter a prompt to use for the refactoring: ")
    output = coordinator.run(file_contents + f"\n\n{prompt}")
    print(output)
