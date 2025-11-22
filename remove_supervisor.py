
#!/usr/bin/env python3
"""
Script to remove supervisor_result and supervisor_explanation fields 
from JSON files for non-detection tasks based on problem_id string matching.
"""

import os
import json
import re
from typing import Dict, Any, List

def is_detection_task(problem_id: str) -> bool:
    """
    Determine if a task is a detection task based on problem_id string matching.
    Detection tasks typically contain words like 'detection', 'detect', 'anomaly', etc.
    """
    detection_patterns = [
        r'detection',
        r'detect'
    ]
    
    problem_id_lower = problem_id.lower()
    
    for pattern in detection_patterns:
        if re.search(pattern, problem_id_lower):
            return True
    
    return False

def remove_supervisor_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove supervisor_result and supervisor_explanation fields from data.get("results").
    """
    modified_data = data.copy()
    
    # Check if results field exists
    if 'results' not in modified_data:
        return modified_data, []
    
    # Get the results object
    results = modified_data['results']
    if not isinstance(results, dict):
        return modified_data, []
    
    fields_to_remove = ['supervisor_result', 'supervisor_explanation']
    removed_fields = []
    
    for field in fields_to_remove:
        if field in results:
            del results[field]
            removed_fields.append(field)
    
    return modified_data, removed_fields

def process_directory(directory_path: str) -> None:
    """
    Process all JSON files in the given directory.
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return
    
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    total_files = len(json_files)
    processed_files = 0
    non_detection_files = 0
    modified_files = 0
    
    print(f"Found {total_files} JSON files to process...")
    
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        
        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_files += 1
            
            # Check if it has problem_id field
            if 'problem_id' not in data:
                print(f"  {json_file}: No problem_id field found, skipping")
                continue
            
            problem_id = data['problem_id']
            
            # Check if it's a detection task
            if is_detection_task(problem_id):
                print(f"  {json_file}: Detection task ({problem_id}), keeping supervisor fields")
                continue
            
            non_detection_files += 1
            
            # Remove supervisor fields for non-detection tasks
            modified_data, removed_fields = remove_supervisor_fields(data)
            
            if removed_fields:
                # Write back the modified data
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(modified_data, f, ensure_ascii=False, indent=2)
                
                modified_files += 1
                print(f"  {json_file}: Non-detection task ({problem_id}), removed {removed_fields}")
            else:
                print(f"  {json_file}: Non-detection task ({problem_id}), no supervisor fields to remove")
        
        except json.JSONDecodeError as e:
            print(f"  {json_file}: Error reading JSON - {e}")
        except Exception as e:
            print(f"  {json_file}: Unexpected error - {e}")
    
    print(f"\n--- Summary ---")
    print(f"Total files processed: {processed_files}")
    print(f"Non-detection tasks: {non_detection_files}")
    print(f"Files modified: {modified_files}")

def main():
    """
    Main function to process directories.
    """
    # List of directories to process
    directories = [
        './openai_gpt-5/0922',
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\n{'='*50}")
            print(f"Processing directory: {directory}")
            print(f"{'='*50}")
            process_directory(directory)
        else:
            print(f"Directory {directory} not found, skipping...")

if __name__ == "__main__":
    main()