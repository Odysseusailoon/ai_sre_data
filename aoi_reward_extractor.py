import os
import json
import math
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from dataclasses import dataclass, asdict
import tiktoken

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================
CONFIG = {
    "INPUT_DIR": "/mnt/c/Users/yangpei/work/ai_sre_data/openai_gpt-5/0922/",  # Input directory containing JSON files
    "OUTPUT_DIR": "./res/openai_gpt-5/0922/",  # Output directory for results
    "API_KEY": "OPENAI_API_KEY",   # Your OpenAI API key

    "MODEL": "gpt-5",  # OpenAI model to use
    "MAX_WORKERS": 1,  # Maximum concurrent workers
    "MAX_RETRIES": 3,  # Maximum retry attempts for API calls
    "RETRY_DELAY": 2,  # Delay between retries (seconds)
    "TEMPERATURE": 1,  # Temperature for OpenAI API (lower = more consistent)
    "MAX_TOKENS_PER_CHUNK": 271000,  # Maximum tokens per chunk for sliding window
    "ENABLE_PLACEHOLDER": True,  # Enable placeholder replacement for dynamic values
}

# Placeholder patterns for common dynamic values
PLACEHOLDER_PATTERNS = {
    # Kubernetes pod/deployment random suffixes (e.g., 65868bcdb5-vg2m9)
    "k8s_pod_suffix": {
        "pattern": r'\b[a-z0-9]{8,10}-[a-z0-9]{5}\b',
        "placeholder": "<POD_SUFFIX>",
        "description": "Kubernetes pod random suffix"
    },
    # Kubernetes replica set suffix (e.g., -65868bcdb5)
    "k8s_replicaset_suffix": {
        "pattern": r'-[a-z0-9]{8,10}(?=\s|"|\)|$)',
        "placeholder": "<REPLICA_SUFFIX>",
        "description": "Kubernetes replica set suffix"
    },
    # UUID patterns
    "uuid": {
        "pattern": r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b',
        "placeholder": "<UUID>",
        "description": "UUID"
    },
    # IP addresses
    "ip_address": {
        "pattern": r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        "placeholder": "<IP_ADDRESS>",
        "description": "IP address"
    },
    # Timestamps (various formats)
    "timestamp_iso": {
        "pattern": r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?',
        "placeholder": "<TIMESTAMP>",
        "description": "ISO format timestamp"
    },
    # Container IDs (docker/containerd)
    "container_id": {
        "pattern": r'\b[a-f0-9]{64}\b|\b[a-f0-9]{12}\b',
        "placeholder": "<CONTAINER_ID>",
        "description": "Container ID"
    },
    # Temporary file names with timestamps
    "temp_filename": {
        "pattern": r'/tmp/[a-zA-Z0-9_-]+\d{10,13}[a-zA-Z0-9_-]*',
        "placeholder": "<TEMP_FILE>",
        "description": "Temporary file with timestamp"
    }
}
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Command:
    """Data class for command information"""
    command: str
    type: str
    importance_score: int
    description: str
    sequence_number: int
    original_command: Optional[str] = None  # Store original before placeholder replacement
    placeholders_used: Optional[List[Dict[str, str]]] = None  # Track placeholders used


@dataclass
class GroundTruth:
    """Data class for ground truth data"""
    problem_id: str
    key_commands: List[Dict[str, Any]]


class PlaceholderHandler:
    """Handle placeholder replacement for dynamic values in commands"""

    def __init__(self, patterns: Dict[str, Dict[str, Any]] = None):
        """
        Initialize placeholder handler

        Args:
            patterns: Dictionary of placeholder patterns
        """
        self.patterns = patterns or PLACEHOLDER_PATTERNS

    def apply_placeholders(self, command: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Apply placeholder replacements to a command

        Args:
            command: Original command string

        Returns:
            Tuple of (processed command, list of replacements made)
        """
        processed_command = command
        replacements = []

        # Apply patterns in order of specificity (longer patterns first)
        for pattern_name, pattern_config in sorted(
                self.patterns.items(),
                key=lambda x: len(x[1]['pattern']),
                reverse=True
        ):
            pattern = pattern_config['pattern']
            placeholder = pattern_config['placeholder']

            # Find all matches
            matches = re.finditer(pattern, processed_command)

            for match in matches:
                original_value = match.group()

                # Skip if this looks like it might be a service name or known identifier
                if self._should_skip_replacement(original_value, processed_command):
                    continue

                # Record the replacement
                replacements.append({
                    "type": pattern_name,
                    "original": original_value,
                    "placeholder": placeholder,
                    "position": match.start()
                })

                # Apply replacement
                processed_command = processed_command[:match.start()] + placeholder + processed_command[match.end():]

        return processed_command, replacements

    def _should_skip_replacement(self, value: str, context: str) -> bool:
        """
        Determine if a value should be skipped from replacement

        Args:
            value: The value to check
            context: The full command context

        Returns:
            True if should skip, False otherwise
        """
        # List of known service names or identifiers that shouldn't be replaced
        skip_patterns = [
            'astronomy-shop',
            'fraud-detection',
            'recommendation',
            'payment',
            'checkout',
            'frontend',
            'backend',
            'database',
            'redis',
            'kafka',
            'nginx'
        ]

        # Check if value is part of a known service name
        for pattern in skip_patterns:
            if pattern in context.lower():
                # Check if this value is immediately after the service name
                pattern_pos = context.lower().find(pattern)
                value_pos = context.find(value)
                if pattern_pos >= 0 and value_pos >= 0:
                    if abs(pattern_pos + len(pattern) - value_pos) < 2:
                        return False  # Don't skip, it's likely a pod suffix

        return False


class AOIRewardFunctionExtractor:
    """
    Extract ground truth data for reinforcement learning reward function
    in SRE domain using OpenAI API
    """

    # Updated system prompt to include placeholder handling
    SYSTEM_PROMPT = """You are an expert in Site Reliability Engineering (SRE) and reinforcement learning. 
Your task is to analyze operational test platform execution cases and extract key commands for building a reinforcement learning ground truth dataset.

## Task Requirements:

### 1. Key Command Identification and Importance Assessment
Identify and extract key execution commands that solve the current problem from the provided JSON data, and score them from 1-10 based on importance:

**Scoring Criteria**:
- **8-10 points**: Core commands that directly solve the problem
- **5-7 points**: Important diagnostic or configuration commands
- **1-4 points**: Auxiliary query or verification commands
- **0 points**: Irrelevant commands (exclude from dataset)

### 2. Command Type Classification

#### Probe Commands (probe_command)
- **Definition**: Read-only operations that do not change system state
- **Typical Operations**: GET, LIST, DESCRIBE, LOGS, etc.
- **Use Cases**: Log queries, status checks, resource monitoring, etc.

#### Execute Commands (execute_command)
- **Definition**: Write operations that modify system state
- **Typical Operations**: CREATE, UPDATE, DELETE, RESTART, PATCH, APPLY, etc.
- **Use Cases**: Service restart, configuration modification, resource deletion, etc.

### 3. Command Extraction Rules
- **Format Requirements**: Extract complete commands including function calls
  - ✅ Correct example: `exec_shell("kubectl get pods -n astronomy-shop")`
  - ❌ Wrong example: `kubectl get pods -n astronomy-shop`
- **Structure Preservation**: Maintain the integrity of API call format and parameter structure

### 4. Dynamic Value Handling
**IMPORTANT**: Keep commands exactly as they appear in the original data. DO NOT replace any values with placeholders.
The system will automatically handle dynamic values (like pod names with random suffixes) during post-processing.

Examples of dynamic values that will be handled automatically:
- Pod suffixes: `fraud-detection-65868bcdb5-vg2m9` (the random part will be replaced later)
- Container IDs: `a3f4b5c6d7e8...`
- Timestamps: `2024-01-01T10:30:00Z`
- IP addresses: `192.168.1.100`

### 5. Execution Order Marking
Number all commands that solve the current problem in logical execution order, reflecting operational dependencies.

### 6. Output Format
Return the result in the following JSON format:
```json
{
  "problem_id": "case_identifier",
  "key_commands": [
    {
      "command": "complete_command_content",
      "type": "probe_command|execute_command",
      "importance_score": 1-10,
      "description": "command_function_description",
      "sequence_number": 1
    }
  ]
}
```

### 7. Quality Assurance
- **Accuracy**: Ensure accurate command classification and scoring
- **Completeness**: Cover all commands that contribute to problem solving
- **Consistency**: Maintain uniform scoring standards

Only include commands with importance_score > 0. Sort commands by sequence_number."""

    USER_PROMPT_TEMPLATE = """Please analyze the following JSON data from an SRE operational test case and extract key commands according to the requirements:

**JSON Data**:
```json
{json_content}
```

**Analysis Requirements**:
1. Identify all commands that contribute to solving the problem
2. Classify each command as either probe_command or execute_command
3. Assign importance scores (1-10) based on their contribution to problem resolution
4. Provide clear descriptions of what each command does
5. Number commands in their logical execution sequence
6. Keep all command strings exactly as they appear (including any dynamic values like pod names)

Return only the JSON formatted ground truth data without any additional explanation."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview",
                 max_retries: int = 3, retry_delay: int = 2, temperature: float = 1,
                 max_tokens_per_chunk: int = 100000, enable_placeholder: bool = True):
        """
        Initialize the extractor with OpenAI API configuration

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4-turbo-preview)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            temperature: Temperature for OpenAI API
            max_tokens_per_chunk: Maximum tokens per chunk for sliding window
            enable_placeholder: Enable placeholder replacement
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.enable_placeholder = enable_placeholder

        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize placeholder handler
        self.placeholder_handler = PlaceholderHandler() if enable_placeholder else None

        logger.info(f"Initialized extractor with model: {self.model}")
        logger.info(f"Max tokens per chunk: {self.max_tokens_per_chunk}")
        logger.info(f"Placeholder replacement: {'Enabled' if enable_placeholder else 'Disabled'}")

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def _split_json_into_chunks(self, json_data: Dict[str, Any]) -> List[str]:
        """
        Split large JSON data into manageable chunks using sliding window

        Args:
            json_data: The JSON data to split

        Returns:
            List of JSON string chunks
        """
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)

        # Check if splitting is needed
        total_tokens = self._count_tokens(json_str)
        if total_tokens <= self.max_tokens_per_chunk:
            return [json_str]

        logger.info(f"JSON data has {total_tokens} tokens, splitting into chunks...")

        # Split by lines
        lines = json_str.split('\n')

        # Calculate optimal number of chunks
        estimated_chunks = math.ceil(total_tokens / self.max_tokens_per_chunk)
        logger.info(f"Estimated number of chunks: {estimated_chunks}")

        chunks = []
        current_chunk = []
        current_tokens = 0

        # Pre-calculate token count for each line to optimize
        line_tokens_list = []
        for line in lines:
            tokens = self._count_tokens(line + '\n')
            line_tokens_list.append(tokens)

        i = 0
        while i < len(lines):
            line = lines[i]
            line_tokens = line_tokens_list[i]

            # If current chunk is empty, always add the line
            if not current_chunk:
                current_chunk.append(line)
                current_tokens = line_tokens
                i += 1
                continue

            # Check if we can add this line without exceeding the limit
            if current_tokens + line_tokens <= self.max_tokens_per_chunk:
                current_chunk.append(line)
                current_tokens += line_tokens
                i += 1
            else:
                # Current chunk is full, save it
                chunk_str = '\n'.join(current_chunk)
                chunks.append(chunk_str)
                logger.debug(
                    f"Created chunk {len(chunks)} with {current_tokens} tokens ({current_tokens / self.max_tokens_per_chunk * 100:.1f}% of max)")

                # Start new chunk with current line
                current_chunk = [line]
                current_tokens = line_tokens
                i += 1

        # Add remaining chunk if not empty
        if current_chunk:
            chunk_str = '\n'.join(current_chunk)
            chunks.append(chunk_str)
            logger.debug(
                f"Created final chunk {len(chunks)} with {current_tokens} tokens ({current_tokens / self.max_tokens_per_chunk * 100:.1f}% of max)")

        # Verify the split
        total_chunks_tokens = sum(self._count_tokens(chunk) for chunk in chunks)
        logger.info(f"Split into {len(chunks)} chunks (total tokens: {total_chunks_tokens}, original: {total_tokens})")

        # Log chunk sizes for debugging
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = self._count_tokens(chunk)
            logger.info(
                f"  Chunk {i}: {chunk_tokens} tokens ({chunk_tokens / self.max_tokens_per_chunk * 100:.1f}% of max)")

        return chunks

    def _merge_chunk_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from multiple chunks

        Args:
            results: List of results from each chunk

        Returns:
            Merged result
        """
        if not results:
            return None

        if len(results) == 1:
            return results[0]

        # Merge all key commands
        merged = {
            "problem_id": results[0].get("problem_id", ""),
            "key_commands": []
        }

        command_set = set()  # To avoid duplicates
        all_commands = []

        for result in results:
            for cmd in result.get("key_commands", []):
                cmd_str = cmd.get("command", "")
                if cmd_str and cmd_str not in command_set:
                    command_set.add(cmd_str)
                    all_commands.append(cmd)

        # Re-number sequences
        all_commands.sort(key=lambda x: (x.get("sequence_number", 0), x.get("importance_score", 0)), reverse=True)
        for i, cmd in enumerate(all_commands, 1):
            cmd["sequence_number"] = i

        merged["key_commands"] = all_commands

        logger.info(f"Merged {len(results)} chunk results into {len(all_commands)} unique commands")
        return merged

    def extract_commands_from_json(self, json_data: Dict[str, Any],
                                   file_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract key commands from a single JSON file using OpenAI API

        Args:
            json_data: The JSON data to analyze
            file_name: Name of the JSON file (used as problem_id)

        Returns:
            Extracted ground truth data or None if failed
        """
        # Split into chunks if necessary
        chunks = self._split_json_into_chunks(json_data)

        if len(chunks) > 1:
            logger.info(f"Processing {file_name} in {len(chunks)} chunks due to size")

        chunk_results = []

        for chunk_idx, chunk in enumerate(chunks):
            # Prepare the user prompt
            user_prompt = self.USER_PROMPT_TEMPLATE.format(json_content=chunk)

            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    logger.debug(
                        f"Processing {file_name} chunk {chunk_idx + 1}/{len(chunks)} (attempt {attempt + 1}/{self.max_retries})")

                    # Call OpenAI API
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        response_format={"type": "json_object"}  # Ensure JSON response
                    )

                    # Parse the response
                    result = json.loads(response.choices[0].message.content)

                    # Add problem_id if not present
                    if "problem_id" not in result:
                        result["problem_id"] = Path(file_name).stem

                    # Apply placeholder replacement if enabled
                    if self.enable_placeholder and self.placeholder_handler:
                        result = self._apply_placeholders_to_result(result)

                    # Validate the result structure
                    if self._validate_result(result):
                        chunk_results.append(result)
                        break
                    else:
                        logger.warning(f"Invalid result structure for {file_name} chunk {chunk_idx + 1}")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for {file_name} chunk {chunk_idx + 1}: {e}")
                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit reached, waiting before retry: {e}")
                    time.sleep(self.retry_delay * 2)
                except openai.APIError as e:
                    if "maximum context length" in str(e).lower() or "tokens" in str(e).lower():
                        logger.error(f"Token limit exceeded for chunk {chunk_idx + 1}, skipping this chunk")
                        break
                    else:
                        logger.error(f"API error for {file_name} chunk {chunk_idx + 1}: {e}")
                except Exception as e:
                    logger.error(f"Error processing {file_name} chunk {chunk_idx + 1} (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)

        # Merge results from all chunks
        if chunk_results:
            merged_result = self._merge_chunk_results(chunk_results)
            logger.info(f"Successfully extracted commands from {file_name}")
            return merged_result
        else:
            logger.error(f"Failed to process any chunks for {file_name}")
            return None

    def _apply_placeholders_to_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply placeholder replacements to extracted commands

        Args:
            result: The extracted result

        Returns:
            Result with placeholders applied
        """
        if not result or "key_commands" not in result:
            return result

        for cmd in result["key_commands"]:
            original_command = cmd.get("command", "")
            if original_command:
                # Store original command
                cmd["original_command"] = original_command

                # Apply placeholders
                processed_command, replacements = self.placeholder_handler.apply_placeholders(original_command)

                cmd["command"] = processed_command
                if replacements:
                    cmd["placeholders_used"] = replacements
                    logger.debug(f"Applied {len(replacements)} placeholders to command: {original_command[:50]}...")

        return result

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate the structure of extracted result

        Args:
            result: The extracted result to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(result, dict):
            return False

        if "key_commands" not in result:
            return False

        if not isinstance(result["key_commands"], list):
            return False

        required_fields = ["command", "type", "importance_score",
                           "description", "sequence_number"]

        for cmd in result["key_commands"]:
            if not all(field in cmd for field in required_fields):
                return False
            if cmd["type"] not in ["probe_command", "execute_command"]:
                return False
            if not (0 <= cmd["importance_score"] <= 10):
                return False

        return True

    def process_directory(self, input_dir: str, output_dir: str,
                          max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Process all JSON files in the specified directory

        Args:
            input_dir: Input directory containing JSON files
            output_dir: Output directory for saving results
            max_workers: Maximum number of concurrent workers

        Returns:
            List of all extracted ground truth data
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get all JSON files
        json_files = list(Path(input_dir).glob("*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {input_dir}")
            return []

        logger.info(f"Found {len(json_files)} JSON files to process")

        results = []
        failed_files = []

        # Process files with progress bar
        with tqdm(total=len(json_files), desc="Processing files") as pbar:
            for json_file in json_files:
                try:
                    # Read JSON file
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

                    # Extract commands
                    ground_truth = self.extract_commands_from_json(
                        json_data, json_file.name
                    )

                    if ground_truth:
                        results.append(ground_truth)

                        # Save individual result
                        output_file = Path(output_dir) / f"{json_file.stem}_ground_truth.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(ground_truth, f, ensure_ascii=False, indent=2)
                    else:
                        failed_files.append(json_file.name)

                except Exception as e:
                    logger.error(f"Failed to process {json_file.name}: {e}")
                    failed_files.append(json_file.name)

                pbar.update(1)

        # Save consolidated results
        if results:
            consolidated_file = Path(output_dir) / "consolidated_ground_truth.json"
            with open(consolidated_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved consolidated results to {consolidated_file}")

        # Save failed files list
        if failed_files:
            failed_files_path = Path(output_dir) / "failed_files.json"
            with open(failed_files_path, 'w', encoding='utf-8') as f:
                json.dump({"failed_files": failed_files, "count": len(failed_files)},
                          f, ensure_ascii=False, indent=2)
            logger.info(f"Saved failed files list to {failed_files_path}")

        # Log summary
        logger.info(f"Processing complete: {len(results)} successful, {len(failed_files)} failed")
        if failed_files:
            logger.warning(f"Failed files: {failed_files}")

        return results

    def generate_statistics(self, results: List[Dict[str, Any]],
                            output_dir: str) -> Dict[str, Any]:
        """
        Generate statistics from the extracted ground truth data

        Args:
            results: List of extracted ground truth data
            output_dir: Directory to save statistics

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_problems": len(results),
            "total_commands": 0,
            "probe_commands": 0,
            "execute_commands": 0,
            "commands_with_placeholders": 0,
            "total_placeholders_used": 0,
            "placeholder_types": {},
            "avg_commands_per_problem": 0,
            "avg_importance_score": 0,
            "importance_score_distribution": {i: 0 for i in range(1, 11)},
            "command_type_distribution": {},
            "problems_by_command_count": {},
            "timestamp": datetime.now().isoformat()
        }

        all_scores = []
        command_counts = []

        for result in results:
            commands = result.get("key_commands", [])
            num_commands = len(commands)
            stats["total_commands"] += num_commands
            command_counts.append(num_commands)

            # Track problems by command count
            if str(num_commands) not in stats["problems_by_command_count"]:
                stats["problems_by_command_count"][str(num_commands)] = 0
            stats["problems_by_command_count"][str(num_commands)] += 1

            for cmd in commands:
                # Count command types
                if cmd["type"] == "probe_command":
                    stats["probe_commands"] += 1
                else:
                    stats["execute_commands"] += 1

                # Collect scores
                score = cmd["importance_score"]
                all_scores.append(score)
                stats["importance_score_distribution"][score] += 1

                # Count placeholder usage
                if "placeholders_used" in cmd and cmd["placeholders_used"]:
                    stats["commands_with_placeholders"] += 1
                    for placeholder in cmd["placeholders_used"]:
                        placeholder_type = placeholder.get("type", "unknown")
                        if placeholder_type not in stats["placeholder_types"]:
                            stats["placeholder_types"][placeholder_type] = 0
                        stats["placeholder_types"][placeholder_type] += 1
                        stats["total_placeholders_used"] += 1

        # Calculate averages
        if stats["total_problems"] > 0:
            stats["avg_commands_per_problem"] = round(stats["total_commands"] / stats["total_problems"], 2)

        if all_scores:
            stats["avg_importance_score"] = round(sum(all_scores) / len(all_scores), 2)
            stats["min_importance_score"] = min(all_scores)
            stats["max_importance_score"] = max(all_scores)

        if command_counts:
            stats["min_commands_per_problem"] = min(command_counts)
            stats["max_commands_per_problem"] = max(command_counts)

        stats["command_type_distribution"] = {
            "probe_command": stats["probe_commands"],
            "execute_command": stats["execute_commands"],
            "probe_percentage": round(stats["probe_commands"] / stats["total_commands"] * 100, 2) if stats[
                                                                                                         "total_commands"] > 0 else 0,
            "execute_percentage": round(stats["execute_commands"] / stats["total_commands"] * 100, 2) if stats[
                                                                                                             "total_commands"] > 0 else 0
        }

        # Add placeholder statistics
        if stats["total_commands"] > 0:
            stats["placeholder_coverage"] = round(
                stats["commands_with_placeholders"] / stats["total_commands"] * 100, 2
            )
        else:
            stats["placeholder_coverage"] = 0

        # Save statistics
        stats_file = Path(output_dir) / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Statistics saved to {stats_file}")

        return stats


def print_summary(stats: Dict[str, Any]):
    """
    Print a formatted summary of the extraction results

    Args:
        stats: Statistics dictionary
    """
    print("\n" + "=" * 60)
    print(" " * 20 + "EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total problems processed:        {stats['total_problems']}")
    print(f"Total commands extracted:        {stats['total_commands']}")
    print("-" * 60)
    print(
        f"Probe commands:                  {stats['probe_commands']} ({stats['command_type_distribution']['probe_percentage']}%)")
    print(
        f"Execute commands:                {stats['execute_commands']} ({stats['command_type_distribution']['execute_percentage']}%)")
    print("-" * 60)
    print(f"Average commands per problem:    {stats['avg_commands_per_problem']}")
    print(f"Min commands per problem:        {stats.get('min_commands_per_problem', 'N/A')}")
    print(f"Max commands per problem:        {stats.get('max_commands_per_problem', 'N/A')}")
    print("-" * 60)
    print(f"Average importance score:        {stats['avg_importance_score']}")
    print(f"Min importance score:            {stats.get('min_importance_score', 'N/A')}")
    print(f"Max importance score:            {stats.get('max_importance_score', 'N/A')}")

    # Add placeholder statistics
    if CONFIG["ENABLE_PLACEHOLDER"]:
        print("-" * 60)
        print(
            f"Commands with placeholders:      {stats.get('commands_with_placeholders', 0)} ({stats.get('placeholder_coverage', 0)}%)")
        print(f"Total placeholders used:         {stats.get('total_placeholders_used', 0)}")
        if stats.get('placeholder_types'):
            print("Placeholder types used:")
            for ptype, count in stats['placeholder_types'].items():
                print(f"  {ptype:25s}: {count}")

    print("-" * 60)
    print("Importance Score Distribution:")
    for score in range(1, 11):
        count = stats['importance_score_distribution'][score]
        if count > 0:
            bar = "█" * (count // 2 if count > 0 else 1)
            print(f"  Score {score:2d}: {count:4d} {bar}")
    print("=" * 60)
    print(f"Timestamp: {stats['timestamp']}")
    print("=" * 60 + "\n")


def main():
    """Main function to run the extraction process"""

    # Validate configuration
    if CONFIG["API_KEY"] == "your-openai-api-key-here":
        logger.error("Please set your OpenAI API key in the CONFIG section")
        return

    # Check if input directory exists
    if not Path(CONFIG["INPUT_DIR"]).exists():
        logger.error(f"Input directory does not exist: {CONFIG['INPUT_DIR']}")
        logger.info(f"Please create the directory and add JSON files, or update the INPUT_DIR in CONFIG")
        return

    print("\n" + "=" * 60)
    print(" " * 15 + "AOI REWARD FUNCTION EXTRACTOR")
    print("=" * 60)
    print(f"Input Directory:  {CONFIG['INPUT_DIR']}")
    print(f"Output Directory: {CONFIG['OUTPUT_DIR']}")
    print(f"Model:            {CONFIG['MODEL']}")
    print(f"Max Workers:      {CONFIG['MAX_WORKERS']}")
    print(f"Max Tokens/Chunk: {CONFIG['MAX_TOKENS_PER_CHUNK']}")
    print(f"Placeholders:     {'Enabled' if CONFIG['ENABLE_PLACEHOLDER'] else 'Disabled'}")
    print("=" * 60 + "\n")

    # Initialize extractor
    extractor = AOIRewardFunctionExtractor(
        api_key=CONFIG["API_KEY"],
        model=CONFIG["MODEL"],
        max_retries=CONFIG["MAX_RETRIES"],
        retry_delay=CONFIG["RETRY_DELAY"],
        temperature=CONFIG["TEMPERATURE"],
        max_tokens_per_chunk=CONFIG["MAX_TOKENS_PER_CHUNK"],
        enable_placeholder=CONFIG["ENABLE_PLACEHOLDER"]
    )

    # Process directory
    results = extractor.process_directory(
        input_dir=CONFIG["INPUT_DIR"],
        output_dir=CONFIG["OUTPUT_DIR"],
        max_workers=CONFIG["MAX_WORKERS"]
    )

    # Generate statistics and print summary
    if results:
        stats = extractor.generate_statistics(results, CONFIG["OUTPUT_DIR"])
        print_summary(stats)

        print(f"✅ Results saved to: {CONFIG['OUTPUT_DIR']}")
        print(f"   - Individual ground truth files: *_ground_truth.json")
        print(f"   - Consolidated results: consolidated_ground_truth.json")
        print(f"   - Statistics: statistics.json")

        # Check for failed files
        failed_files_path = Path(CONFIG["OUTPUT_DIR"]) / "failed_files.json"
        if failed_files_path.exists():
            print(f"   - Failed files list: failed_files.json")
    else:
        print("❌ No results extracted. Please check the logs for errors.")


if __name__ == "__main__":
    # Install required packages if not available
    try:
        import tiktoken
    except ImportError:
        print("Installing required package: tiktoken")
        import subprocess

        subprocess.check_call(["pip", "install", "tiktoken"])
        import tiktoken

    main()