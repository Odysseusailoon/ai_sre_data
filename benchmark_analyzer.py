#!/usr/bin/env python3
"""
Benchmark Results Analyzer
Analyzes JSON result files from benchmark runs to generate detailed and summary CSV reports.
"""

import json
import os
import csv
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict


def extract_task_type(problem_id):
    """Extract task type from problem_id"""
    if not problem_id:
        return 'unknown'

    # Extract task type using regex - looking for patterns like 'detection', 'localization', 'analysis', 'mitigation'
    task_types = ['detection', 'localization', 'analysis', 'mitigation']

    for task_type in task_types:
        if task_type in problem_id.lower():
            return task_type

    return 'unknown'

def analyze_json_files(input_dir, is_supervisor_enabled):
    """Analyze all JSON files in the input directory and return detailed results"""
    results = []
    json_dir = Path(input_dir)
    
    # Find all JSON files recursively
    json_files = list(json_dir.glob("**/*.json"))
    
    print(f"Found {len(json_files)} JSON files to analyze...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Extract key information
            result = {
                'filename': json_file.name,
                'session_id': data.get('session_id', ''),
                'problem_id': data.get('problem_id', ''),
                'success': None,
                'task_type': extract_task_type(data.get('problem_id', '')),
                'in_tokens': data.get('results', {}).get('in_tokens', None),
                'out_tokens': data.get('results', {}).get('out_tokens', None),
                'start_time': data.get('start_time', None),
                'end_time': data.get('end_time', None),
                'duration': None,
                'steps': data.get('results', {}).get('steps', None),
                'supervisor_result': data.get('results', {}).get('supervisor_result', None),
            }
            
            # Handle different success indicators (reference enhanced_analysis logic)
            results_data = data.get('results', {})
            
            # Check various success fields
            if 'success' in results_data:
                result['success'] = results_data.get('success', False)
            elif 'Detection Accuracy' in results_data:
                if is_supervisor_enabled:
                    result['success'] = results_data.get('Detection Accuracy') == 'Correct' and result['supervisor_result'] == 'Correct'
                else:
                    result['success'] = results_data.get('Detection Accuracy') == 'Correct'
            elif 'Localization Accuracy' in results_data:
                result['success'] = results_data.get('Localization Accuracy') == 'Correct'
            elif 'Analysis Accuracy' in results_data:
                result['success'] = results_data.get('Analysis Accuracy') == 'Correct'
            elif 'Mitigation Accuracy' in results_data:
                result['success'] = results_data.get('Mitigation Accuracy') == 'Correct'
            elif 'Accuracy' in results_data:
                result['success'] = results_data.get('Accuracy') == 'Correct'


            # Calculate duration if not available in TTM
            if result['start_time'] and result['end_time']:
                result['duration'] = result['end_time'] - result['start_time']
            
            # Convert timestamps to Beijing time
            result['start_time'] = convert_timestamp_to_utc(result['start_time'])
            result['end_time'] = convert_timestamp_to_utc(result['end_time'])
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return results


def generate_detailed_csv(results, output_dir):
    """Generate detailed CSV with all results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_csv = output_path / f"detailed_results_{timestamp}.csv"
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(detailed_csv, index=False)
    
    print(f"Detailed results written to {detailed_csv}\n")
    return df, detailed_csv


def generate_observation_csv(df, output_dir):
    """Generate observation CSV with task type statistics"""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    observation_csv = output_path / f"observation_results_{timestamp}.csv"
    
    # Initialize results for 4 task types
    task_types = ['detection', 'localization', 'analysis', 'mitigation']
    observation_results = []
    
    for task_type in task_types:
        task_data = df[df['task_type'] == task_type]
        
        if len(task_data) == 0:
            # If no data for this task type, add empty row
            observation_results.append({
                'taskType': task_type,
                'successRate': 0.0,
                'averageTime': 0.0,
                'averageSteps': 0.0,
                'longestTime': 0.0,
                'mostSteps': 0,
                'leastTime': 0.0,
                'leastSteps': 0
            })
            continue
        
        # Success rate calculation
        total_tasks = len(task_data)
        successful_tasks = len(task_data[task_data['success'] == True])
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Time metrics (TTM for all tasks, including failed ones where available)
        time_values = task_data['duration'].dropna()
        avg_time = time_values.mean() if len(time_values) > 0 else 0
        longest_time = time_values.max() if len(time_values) > 0 else 0
        least_time = time_values.min() if len(time_values) > 0 else 0
        
        # Step metrics
        step_values = task_data['steps'].dropna()
        avg_steps = step_values.mean() if len(step_values) > 0 else 0
        most_steps = step_values.max() if len(step_values) > 0 else 0
        least_steps = step_values.min() if len(step_values) > 0 else 0
        
        observation_results.append({
            'taskType': task_type,
            'successRate': round(success_rate, 2),
            'averageTime': round(avg_time, 2),
            'averageSteps': round(avg_steps, 2),
            'longestTime': round(longest_time, 2),
            'mostSteps': int(most_steps) if pd.notna(most_steps) else 0,
            'leastTime': round(least_time, 2),
            'leastSteps': int(least_steps) if pd.notna(least_steps) else 0
        })
    
    # Convert to DataFrame and save
    observation_df = pd.DataFrame(observation_results)
    observation_df.to_csv(observation_csv, index=False)
    
    print(f"Observation results written to {observation_csv}\n")
    return observation_df, observation_csv

def convert_timestamp_to_utc(timestamp):
    """Convert Unix timestamp to Beijing time (UTC+8) string"""
    if timestamp is None:
        return None

    # Create datetime object from timestamp (assuming UTC)
    dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    # Format as YYYY-MM-DD HH:MM:SS
    return dt_utc.strftime('%Y-%m-%d %H:%M:%S')

def print_summary_stats(df):
    """Print summary statistics to console"""
    print("## BENCHMARK ANALYSIS SUMMARY")
    
    # Overall statistics
    total_tasks = len(df)
    successful_tasks = len(df[df['success'] == True])
    failed_tasks = len(df[df['success'] == False])
    unknown_tasks = len(df[df['success'].isna()])
    
    print(f"\n**OVERALL STATISTICS:**")
    print(f"- Total Tasks: {total_tasks}")
    print(f"- Successful Tasks: {successful_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
    print(f"- Failed Tasks: {failed_tasks} ({failed_tasks/total_tasks*100:.1f}%)")
    print(f"- Unknown Status: {unknown_tasks} ({unknown_tasks/total_tasks*100:.1f}%)")
    
    # Task type distribution
    print(f"\n**TASK TYPE DISTRIBUTION:**")
    task_counts = df['task_type'].value_counts()
    for task_type, count in task_counts.items():
        task_success = len(df[(df['task_type'] == task_type) & (df['success'] == True)])
        task_success_rate = (task_success / count * 100) if count > 0 else 0
        print(f"- {task_type.capitalize()}: {count} tasks, {task_success_rate:.1f}% success rate")


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results and generate CSV reports')
    parser.add_argument('--input-dir', '-i',
                       default=str(Path.cwd()),
                       help='Input directory containing JSON result files (default: current directory)')
    parser.add_argument('--output-dir', '-o',
                       default=str(Path.cwd()),
                       help='Output directory for CSV reports (default: current directory)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress console summary output')
    parser.add_argument('--supervisor', '-s', help="filter result by supervisor_result", default=True)
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return 1
    
    print(f"Analyzing JSON files in: {args.input_dir}\n")
    print(f"Output directory: {args.output_dir}\n")
    
    # Analyze JSON files
    results = analyze_json_files(args.input_dir, args.supervisor)
    
    if not results:
        print("No valid JSON files found to analyze.")
        return 1
    
    # Generate detailed CSV
    df, detailed_csv_path = generate_detailed_csv(results, args.output_dir)
    
    # Generate observation CSV
    observation_df, observation_csv_path = generate_observation_csv(df, args.output_dir)
    
    # Print summary unless quiet mode
    if not args.quiet:
        print_summary_stats(df)

    print("## GENERATED FILES:")
    print(f"1. Detailed Results: {detailed_csv_path}")
    print(f"2. Observation Results: {observation_csv_path}")
    print(f"\nAnalysis complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())