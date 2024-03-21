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

    def send_message(self, messages):
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_message}] + messages,
            temperature=0,
            stream=True                                                                   # Add this line to enable streaming
        )

        output = ""
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is not None:
                output += token
                sys.stdout.write(token)
                sys.stdout.flush()
            else:
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
        coordinator_output = self.coordinator_agent.send_message(messages)
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

    refactor_coordinator = Agent(system_message="As the principal orchestrator, your expertise lies in directing the goal workflow across the agents `refactor_runner`, `refactor_suggestor`, `refactor_editor`. Your goal is returning a fully refactored code from a source code file or codebase provided by the user. For this, you'll coordinate your assistants to perform their duties, one at a time, and keep track of their responses.\n\nYour operational blueprint must follow the following steps.\nWARNING: You MUST speak with `refactor_runner` before speaking/choosing any other agent.\n\nREFACTORING:\n1. If the user gave a path as input, ask `refactor_runner` to write some code to read and print it. If the user presented it in text form already, ask `refactor_runner` to save it's contents into a file on the disk.\n2. Present the source code in language format to the `refactor_suggestor`, whom will reply with required refactoring task that needs to be performed.\n3. For each task, present the latest version of the code alongside the task to the `refactor_editor` assistant. The assistant will create .patch files to be applied on top of the input files.\n4. Ask `refactor_runner` to execute the required shell commands (using `sh` codeblocks) that will apply the patches on top of the source code file(s). `refactor_runner` might require several consecutive runs. If it reaches 10 or replies with CONTINUE, it means it's job is done and you can proceed to step 4. If it replies with TERMINATE, stop everything.\n5. Repeat steps 2-4 until `refactor_suggestor` doesn't have any more tasks.",
                            name="refactor_coordinator")
    refactor_suggestor = Agent(system_message="You are an intelligent agent focused on identifying the impactful structural improvements in a given piece of code. Use your development skills to derive a direct straightforward tasks that could be performed for improving the overall code, including legibility, good practices, performance and others that you consider worthy of refactoring. Do not continue if you don't have access to the source code. Don't ask for too abstract or general questions like '- Remove all synchronous code, specifically the blocking I/O operations, and refactor them to be asynchronous'. Instead, be specific and tackle it one at a time, like '- Remove all synchronous code from myFunction and make them asynchronous', then afterwards '- Replace blocking I/O operations with their asynchronous versions', and so on. Each response must contain exclusively one single task. Focus on big structural changes. Do not remove comments or waste time cleaning up unused things.\n\nAn example follows below:\nInput:\n[very long partially refactored code]\nOutput:\n- Split `callDatabase` into two separate functions adhering to SRP: `connectDatabase`, `retrieveDatabaseData`\n\nInput:\n[very long code]\nOutput:\n- Simplify function `getMyModals`\n\nInput:\n[very long even more refactored code]\nOutput:\n- Join functions `getMyStack`, `getCurrentPatch` and `getLatestData` into a single class called `Modal`, in a new file. Use getters for those variables.\n\nDO NOT return anything besides the task, like in the example below, as your response will be programatically parsed. NEVER TRY OR ASK TO RUN CODE OR FUNCTIONS. NEVER WORK WITH HYPOTHETICAL EXAMPLES OR CODE. Start with the tasks that have the most significant structural impact, like requiring to create files or splitting chunks of code.\n",
                               name="refactor_suggestor")
    refactor_editor = Agent(system_message="Your task is to meticulously create patch files that edit an input code ensuring the provided task is performed. You will do this by saving patch files with the diffs containing revised code as per the task indicates, BUT IN THE FORMAT OF A PATCHFILE (.patch). You will use shell commands (a sh codeblock) to save these patches to the disk. When writing the patch files be as succint as possible, and try your best to make sure those patches will apply without errors onto the input file. Verify each patch yourself. For long contexts of text, split chunks that would cause summarization of inner contents into several separate patches/codeblocks instead. DO NOT summarize or omit sections that are within the middle part of each code block, split those into separate chunks instead. If the input files do not exist, create them by dumping their contents into a file first. NEVER WORK WITH HYPOTHETICAL EXAMPLES OR CODE.",
                            name="refactor_editor")

    coordinator = Coordinator(refactor_coordinator, agents=[refactor_suggestor, refactor_editor])

    file_contents = get_file_contents(file_paths)
    prompt = input("Enter a prompt to use for the refactoring: ")
    output = coordinator.run(file_contents + f"\n\n{prompt}")
    print(output)
