import re


def read_content_until_delimiter(lines, start_index):
    """Read content from start_index until next ===== line or end of file"""
    content_lines = []
    i = start_index

    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("====="):
            break
        content_lines.append(line)
        i += 1

    content = "\n".join(content_lines).strip()
    return content if content else ""


# Regular expressions for parsing
PROMPT_PATTERN = re.compile(r"^===== prompt =====$")
AGENT_PATTERN = re.compile(r"^===== Agent")
CODE_BLOCK_PATTERN = re.compile(r"^```\n(.*?)\n```$", re.DOTALL)
EXEC_SHELL_PATTERN = re.compile(r"exec_shell\([\"'](.*?)[\"']\)")


def processing_file(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if PROMPT_PATTERN.match(line):
                # Read prompt content until next delimiter
                prompt = read_content_until_delimiter(lines, i + 1)

                # Look for the corresponding Agent response
                j = i + 1
                while j < len(lines):
                    if AGENT_PATTERN.match(lines[j].strip()):
                        # Read agent response content
                        raw_response = read_content_until_delimiter(lines, j + 1)

                        # Extract content from markdown code blocks using regex
                        code_match = CODE_BLOCK_PATTERN.match(raw_response)
                        if code_match:
                            response = code_match.group(1).strip()
                        else:
                            response = raw_response

                        # Remove exec_shell() wrapper using regex
                        exec_match = EXEC_SHELL_PATTERN.match(response)
                        if exec_match:
                            response = exec_match.group(1)

                        # Create conversation pair
                        conversation = {
                            'prompt': prompt,
                            'response': response
                        }
                        res.append(conversation)
                        break
                    j += 1
            i += 1

    return res


if __name__ == "__main__":
    import os
    import json

    # Process all .txt files in the openai_gpt-5 directory
    path =  './openai_gpt-5/0922'
    txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for txt_file in txt_files:
        print(f"Processing file: {txt_file}")
        txt_path = os.path.join(path, txt_file)

        # Extract conversations from the txt file
        conversations = processing_file(txt_path)

        # Create corresponding JSON filename
        json_filename = txt_file.replace('.txt', '.json')
        json_path = os.path.join(path, json_filename)

        # Read existing JSON file if it exists
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}

        # Add conversations field
        data['conversation'] = conversations

        # Write back to JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Added {len(conversations)} conversations to {json_filename}")