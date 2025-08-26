#!/usr/bin/env python3
"""
AIOpsLab Benchmark Results Statistics Analyzer
Analyzes JSON result files from benchmark runs to generate statistics CSV.
"""

import json
import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import re


def extract_problem_info(problem_id):
    """Extract problem type and task type from problem_id"""
    # Pattern: problem_name-task_type-variant
    parts = problem_id.split('-')
    if len(parts) >= 2:
        task_type = parts[-2]  # detection, localization, analysis, mitigation
        problem_base = '-'.join(parts[:-2])  # everything before task_type-variant
        return problem_base, task_type
    return problem_id, "unknown"


def categorize_problem_type(problem_base):
    """Categorize problems into major types"""
    categories = {
        'infrastructure': [
            'k8s_target_port', 'auth_miss_mongodb', 'revoke_auth_mongodb',
            'storage_user_unregistered', 'redeploy_without_pv', 'network_loss',
            'network_delay', 'kernel_fault', 'disk_woreout'
        ],
        'application': [
            'misconfig_app', 'ad_service_failure', 'ad_service_high_cpu',
            'ad_service_manual_gc', 'cart_service_failure', 'payment_service_failure',
            'payment_service_unreachable', 'product_catalog_failure',
            'recommendation_service_cache_failure', 'image_slow_load',
            'loadgenerator_flood_homepage', 'kafka_queue_problems',
            'flower_model_misconfig'
        ],
        'operational': [
            'scale_pod', 'assign_non_existent_node', 'container_kill',
            'pod_failure', 'pod_kill', 'wrong_bin_usage', 'operator_misoperation',
            'flower_node_stop'
        ],
        'baseline': ['no_op']
    }

    for category, problems in categories.items():
        for problem in problems:
            if problem in problem_base:
                return category
    return 'other'


def analyze_results(results_dir):
    """Analyze all JSON result files in the directory"""
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return None

    json_files = list(results_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON result files")

    if not json_files:
        print("No JSON files found in the results directory")
        return None

    # Statistics containers
    stats = {
        'total_cases': 0,
        'successful_cases': 0,
        'failed_cases': 0,
        'by_agent': defaultdict(
            lambda: {'total': 0, 'success': 0, 'failed': 0, 'tokens_in': 0, 'tokens_out': 0, 'total_time': 0}),
        'by_problem_type': defaultdict(
            lambda: {'total': 0, 'success': 0, 'failed': 0, 'tokens_in': 0, 'tokens_out': 0, 'total_time': 0}),
        'by_task_type': defaultdict(
            lambda: {'total': 0, 'success': 0, 'failed': 0, 'tokens_in': 0, 'tokens_out': 0, 'total_time': 0}),
        'by_category': defaultdict(
            lambda: {'total': 0, 'success': 0, 'failed': 0, 'tokens_in': 0, 'tokens_out': 0, 'total_time': 0}),
        'detailed_cases': []
    }

    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract basic information
            session_id = data.get('session_id', 'unknown')
            agent = data.get('agent', 'unknown')
            problem_id = data.get('problem_id', 'unknown')
            start_time = data.get('start_time', 0)
            end_time = data.get('end_time', 0)

            # Extract results
            results = data.get('results', {})

            # Handle different success indicators
            success = False
            success_field = None
            success_value = None

            if 'success' in results:
                success_field = 'success'
                success_value = results.get('success', False)
                success = success_value
            elif 'Detection Accuracy' in results:
                success_field = 'Detection Accuracy'
                success_value = results.get('Detection Accuracy')
                success = success_value == 'Correct'
            elif 'Localization Accuracy' in results:
                success_field = 'Localization Accuracy'
                success_value = results.get('Localization Accuracy')
                success = success_value == 'Correct'
            elif 'Analysis Accuracy' in results:
                success_field = 'Analysis Accuracy'
                success_value = results.get('Analysis Accuracy')
                success = success_value == 'Correct'
            elif 'Mitigation Accuracy' in results:
                success_field = 'Mitigation Accuracy'
                success_value = results.get('Mitigation Accuracy')
                success = success_value == 'Correct'
            elif 'Accuracy' in results:
                success_field = 'Accuracy'
                success_value = results.get('Accuracy')
                success = success_value == 'Correct'

            # Handle different time metrics
            ttm = 0
            if 'TTM' in results:
                ttm = results.get('TTM', 0)  # Time to Mitigate
            elif 'TTD' in results:
                ttm = results.get('TTD', 0)  # Time to Detect
            elif 'TTL' in results:
                ttm = results.get('TTL', 0)  # Time to Localize
            elif 'TTA' in results:
                ttm = results.get('TTA', 0)  # Time to Analyze

            steps = results.get('steps', 0)
            in_tokens = results.get('in_tokens', 0)
            out_tokens = results.get('out_tokens', 0)

            # Calculate execution time
            execution_time = end_time - start_time if end_time > start_time else ttm

            # Extract problem information
            problem_base, task_type = extract_problem_info(problem_id)
            category = categorize_problem_type(problem_base)

            # Update statistics
            stats['total_cases'] += 1
            if success:
                stats['successful_cases'] += 1
            else:
                stats['failed_cases'] += 1

            # Update by agent
            agent_stats = stats['by_agent'][agent]
            agent_stats['total'] += 1
            if success:
                agent_stats['success'] += 1
            else:
                agent_stats['failed'] += 1
            agent_stats['tokens_in'] += in_tokens
            agent_stats['tokens_out'] += out_tokens
            agent_stats['total_time'] += execution_time

            # Update by problem type
            problem_stats = stats['by_problem_type'][problem_base]
            problem_stats['total'] += 1
            if success:
                problem_stats['success'] += 1
            else:
                problem_stats['failed'] += 1
            problem_stats['tokens_in'] += in_tokens
            problem_stats['tokens_out'] += out_tokens
            problem_stats['total_time'] += execution_time

            # Update by task type
            task_stats = stats['by_task_type'][task_type]
            task_stats['total'] += 1
            if success:
                task_stats['success'] += 1
            else:
                task_stats['failed'] += 1
            task_stats['tokens_in'] += in_tokens
            task_stats['tokens_out'] += out_tokens
            task_stats['total_time'] += execution_time

            # Update by category
            category_stats = stats['by_category'][category]
            category_stats['total'] += 1
            if success:
                category_stats['success'] += 1
            else:
                category_stats['failed'] += 1
            category_stats['tokens_in'] += in_tokens
            category_stats['tokens_out'] += out_tokens
            category_stats['total_time'] += execution_time

            # Store detailed case information
            case_detail = {
                'session_id': session_id,
                'agent': agent,
                'problem_id': problem_id,
                'problem_base': problem_base,
                'task_type': task_type,
                'category': category,
                'success': success,
                'execution_time': execution_time,
                'task_time': ttm,  # Generic task time (TTD/TTL/TTA/TTM)
                'steps': steps,
                'in_tokens': in_tokens,
                'out_tokens': out_tokens,
                'total_tokens': in_tokens + out_tokens,
                'file_path': str(json_file)
            }
            stats['detailed_cases'].append(case_detail)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return stats


def generate_csv_reports(stats, output_dir):
    """Generate multiple CSV reports from the statistics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Summary Report
    summary_file = output_dir / f"benchmark_summary_{timestamp}.csv"
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Cases', stats['total_cases']])
        writer.writerow(['Successful Cases', stats['successful_cases']])
        writer.writerow(['Failed Cases', stats['failed_cases']])
        writer.writerow(['Success Rate (%)', f"{(stats['successful_cases'] / stats['total_cases'] * 100):.2f}" if stats[
                                                                                                                      'total_cases'] > 0 else "0.00"])

        total_tokens_in = sum(data['tokens_in'] for data in stats['by_agent'].values())
        total_tokens_out = sum(data['tokens_out'] for data in stats['by_agent'].values())
        total_time = sum(data['total_time'] for data in stats['by_agent'].values())

        writer.writerow(['Total Input Tokens', total_tokens_in])
        writer.writerow(['Total Output Tokens', total_tokens_out])
        writer.writerow(['Total Tokens', total_tokens_in + total_tokens_out])
        writer.writerow(['Total Execution Time (s)', f"{total_time:.2f}"])
        writer.writerow(['Average Execution Time (s)',
                         f"{total_time / stats['total_cases']:.2f}" if stats['total_cases'] > 0 else "0.00"])

    # 2. Agent Performance Report
    agent_file = output_dir / f"agent_performance_{timestamp}.csv"
    with open(agent_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Agent', 'Total Cases', 'Successful', 'Failed', 'Success Rate (%)',
                         'Total Input Tokens', 'Total Output Tokens', 'Total Tokens',
                         'Avg Input Tokens', 'Avg Output Tokens', 'Avg Total Tokens',
                         'Total Time (s)', 'Avg Time (s)'])

        for agent, data in sorted(stats['by_agent'].items()):
            success_rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
            avg_in = data['tokens_in'] / data['total'] if data['total'] > 0 else 0
            avg_out = data['tokens_out'] / data['total'] if data['total'] > 0 else 0
            avg_total = (data['tokens_in'] + data['tokens_out']) / data['total'] if data['total'] > 0 else 0
            avg_time = data['total_time'] / data['total'] if data['total'] > 0 else 0

            writer.writerow([agent, data['total'], data['success'], data['failed'],
                             f"{success_rate:.2f}", data['tokens_in'], data['tokens_out'],
                             data['tokens_in'] + data['tokens_out'],
                             f"{avg_in:.1f}", f"{avg_out:.1f}", f"{avg_total:.1f}",
                             f"{data['total_time']:.2f}", f"{avg_time:.2f}"])

    # 3. Problem Type Performance Report
    problem_file = output_dir / f"problem_performance_{timestamp}.csv"
    with open(problem_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Problem Type', 'Category', 'Total Cases', 'Successful', 'Failed', 'Success Rate (%)',
                         'Total Input Tokens', 'Total Output Tokens', 'Total Tokens',
                         'Avg Input Tokens', 'Avg Output Tokens', 'Avg Total Tokens',
                         'Total Time (s)', 'Avg Time (s)'])

        for problem, data in sorted(stats['by_problem_type'].items()):
            category = categorize_problem_type(problem)
            success_rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
            avg_in = data['tokens_in'] / data['total'] if data['total'] > 0 else 0
            avg_out = data['tokens_out'] / data['total'] if data['total'] > 0 else 0
            avg_total = (data['tokens_in'] + data['tokens_out']) / data['total'] if data['total'] > 0 else 0
            avg_time = data['total_time'] / data['total'] if data['total'] > 0 else 0

            writer.writerow([problem, category, data['total'], data['success'], data['failed'],
                             f"{success_rate:.2f}", data['tokens_in'], data['tokens_out'],
                             data['tokens_in'] + data['tokens_out'],
                             f"{avg_in:.1f}", f"{avg_out:.1f}", f"{avg_total:.1f}",
                             f"{data['total_time']:.2f}", f"{avg_time:.2f}"])

    # 4. Task Type Performance Report
    task_file = output_dir / f"task_performance_{timestamp}.csv"
    with open(task_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Task Type', 'Total Cases', 'Successful', 'Failed', 'Success Rate (%)',
                         'Total Input Tokens', 'Total Output Tokens', 'Total Tokens',
                         'Avg Input Tokens', 'Avg Output Tokens', 'Avg Total Tokens',
                         'Total Time (s)', 'Avg Time (s)'])

        for task, data in sorted(stats['by_task_type'].items()):
            success_rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
            avg_in = data['tokens_in'] / data['total'] if data['total'] > 0 else 0
            avg_out = data['tokens_out'] / data['total'] if data['total'] > 0 else 0
            avg_total = (data['tokens_in'] + data['tokens_out']) / data['total'] if data['total'] > 0 else 0
            avg_time = data['total_time'] / data['total'] if data['total'] > 0 else 0

            writer.writerow([task, data['total'], data['success'], data['failed'],
                             f"{success_rate:.2f}", data['tokens_in'], data['tokens_out'],
                             data['tokens_in'] + data['tokens_out'],
                             f"{avg_in:.1f}", f"{avg_out:.1f}", f"{avg_total:.1f}",
                             f"{data['total_time']:.2f}", f"{avg_time:.2f}"])

    # 5. Category Performance Report
    category_file = output_dir / f"category_performance_{timestamp}.csv"
    with open(category_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Total Cases', 'Successful', 'Failed', 'Success Rate (%)',
                         'Total Input Tokens', 'Total Output Tokens', 'Total Tokens',
                         'Avg Input Tokens', 'Avg Output Tokens', 'Avg Total Tokens',
                         'Total Time (s)', 'Avg Time (s)'])

        for category, data in sorted(stats['by_category'].items()):
            success_rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
            avg_in = data['tokens_in'] / data['total'] if data['total'] > 0 else 0
            avg_out = data['tokens_out'] / data['total'] if data['total'] > 0 else 0
            avg_total = (data['tokens_in'] + data['tokens_out']) / data['total'] if data['total'] > 0 else 0
            avg_time = data['total_time'] / data['total'] if data['total'] > 0 else 0

            writer.writerow([category, data['total'], data['success'], data['failed'],
                             f"{success_rate:.2f}", data['tokens_in'], data['tokens_out'],
                             data['tokens_in'] + data['tokens_out'],
                             f"{avg_in:.1f}", f"{avg_out:.1f}", f"{avg_total:.1f}",
                             f"{data['total_time']:.2f}", f"{avg_time:.2f}"])

    # 6. Detailed Cases Report
    details_file = output_dir / f"detailed_cases_{timestamp}.csv"
    with open(details_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Session ID', 'Agent', 'Problem ID', 'Problem Base', 'Task Type', 'Category',
                         'Success', 'Execution Time (s)', 'Task Time', 'Steps', 'Input Tokens', 'Output Tokens',
                         'Total Tokens', 'File Path'])

        # Sort by success status (failed first) then by problem_id
        sorted_cases = sorted(stats['detailed_cases'], key=lambda x: (x['success'], x['problem_id']))

        for case in sorted_cases:
            writer.writerow([
                case['session_id'], case['agent'], case['problem_id'], case['problem_base'],
                case['task_type'], case['category'], case['success'],
                f"{case['execution_time']:.2f}", f"{case['task_time']:.2f}", case['steps'],
                case['in_tokens'], case['out_tokens'], case['total_tokens'], case['file_path']
            ])

    return {
        'summary': summary_file,
        'agents': agent_file,
        'problems': problem_file,
        'tasks': task_file,
        'categories': category_file,
        'details': details_file
    }


def print_summary_stats(stats):
    """Print a summary of the statistics to console"""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    total = stats['total_cases']
    success = stats['successful_cases']
    failed = stats['failed_cases']
    success_rate = (success / total * 100) if total > 0 else 0

    print(f"Total Cases: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {success_rate:.2f}%")

    # Token consumption
    total_tokens_in = sum(data['tokens_in'] for data in stats['by_agent'].values())
    total_tokens_out = sum(data['tokens_out'] for data in stats['by_agent'].values())
    total_tokens = total_tokens_in + total_tokens_out

    print(f"\nToken Consumption:")
    print(f"Input Tokens: {total_tokens_in:,}")
    print(f"Output Tokens: {total_tokens_out:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Avg Tokens per Case: {total_tokens / total:.1f}" if total > 0 else "Avg Tokens per Case: 0.0")

    # Time statistics
    total_time = sum(data['total_time'] for data in stats['by_agent'].values())
    print(f"\nExecution Time:")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {total_time / total:.2f}s" if total > 0 else "Average Time: 0.00s")

    # Agent performance
    print(f"\nAgent Performance:")
    for agent, data in sorted(stats['by_agent'].items()):
        agent_success_rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"  {agent}: {data['success']}/{data['total']} ({agent_success_rate:.1f}%)")

    # Task type performance
    print(f"\nTask Type Performance:")
    for task, data in sorted(stats['by_task_type'].items()):
        task_success_rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"  {task}: {data['success']}/{data['total']} ({task_success_rate:.1f}%)")

    # Category performance
    print(f"\nCategory Performance:")
    for category, data in sorted(stats['by_category'].items()):
        category_success_rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"  {category}: {data['success']}/{data['total']} ({category_success_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze AIOpsLab benchmark results')
    parser.add_argument('--results-dir', '-r',
                        default='aiopslab/data/results',
                        help='Directory containing JSON result files (default: aiopslab/data/results)')
    parser.add_argument('--output-dir', '-o',
                        default='benchmark_analysis',
                        help='Output directory for CSV reports (default: benchmark_analysis)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress console output')

    args = parser.parse_args()

    # Analyze results
    stats = analyze_results(args.results_dir)
    if stats is None:
        return 1

    # Print summary to console unless quiet
    if not args.quiet:
        print_summary_stats(stats)

    # Generate CSV reports
    output_files = generate_csv_reports(stats, args.output_dir)

    print(f"\nGenerated CSV reports:")
    for report_type, file_path in output_files.items():
        print(f"  {report_type}: {file_path}")

    print(f"\nAll reports saved to: {args.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())