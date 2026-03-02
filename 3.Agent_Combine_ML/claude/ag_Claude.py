#!/usr/bin/env python3
import pandas as pd
import subprocess
import multiprocessing as mp
import argparse
import re
import json
from json import loads
from tqdm import tqdm

def predict_icd10_for_note(discharge_note,predicted_codes):
    """Predict ICD-10 codes for a single discharge note using Claude Code"""

    system_content = """You are an expert medical coding specialist. Review predicted ICD-10 codes from machine learning output and keep only those supported by the discharge summary.
    
INSTRUCTIONS:
1. Analyze the discharge summary
2. Review each predicted ICD-10 code
3. Keep codes with clear evidence in the summary
4. Remove unsupported or irrelevant codes

OUTPUT FORMAT (JSON):
{
"relevant_codes": {
    "E11.9": "Type 2 diabetes without complications (0.95)",
    "I25.9": "Chronic ischemic heart disease (0.87)"
},
"removed_codes": ["N18.6", "J44.1"],
"total_kept": 2,
"total_removed": 2
}
"""

    user_content = f"""
    Discharge Summary:
    {discharge_note}
    
    ML-Predicted Codes:
    {predicted_codes}

    Filter and keep only relevant codes:

    
    """

    # Format as chat template
    prompt = f"System: {system_content}\n\nUser: {user_content}\n\nAssistant:"

    try:
        # Use claude with stdin input
        result = subprocess.run(
            ['claude'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30
        )
      
        if result.returncode == 0:
            return result.stdout.strip()
        else:
           # Build dict shaped like CompletedProcess
            fake_cp = {
                "relevant_codes": {
                    "args": ["claude"],
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            }
            
            # Return as JSON string
            return json.dumps(fake_cp)

    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_json_and_codes(prediction_text):
    """Extract JSON content and ICD codes from prediction text"""
    # Extract JSON
   
    match = re.search(r'"relevant_codes":\s*(\{[^}]*\})', prediction_text, re.DOTALL)
    if match:
        try:
            json_str = match.group(1)
            parsed_json = loads(json_str)
            keys = list(parsed_json)
            # Return the JSON as formatted string and keys as array
            formatted_json = json.dumps(parsed_json, indent=2)
            keys_array = json.dumps(keys)
            
            return formatted_json, keys_array
        except:
            return "INVALID_JSON", "[]"
    else:
        return "NO_JSON_FOUND", "[]"

def process_row(row_data):
    """Process a single row for multiprocessing"""
    index, discharge_note, predicted_codes = row_data
    try:
        prediction = predict_icd10_for_note(discharge_note,predicted_codes)
        json_content, icd_codes = extract_json_and_codes(prediction)
        return index, json_content, icd_codes
    except Exception as e:
        return index, f"Error: {str(e)}", "[]"

def main(input_file='Claude.csv', output_file='Claude.csv', num_processes=None, start_row=None, stop_row=None):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Handle row range selection
        if start_row is not None:
            start_idx = start_row - 1  # Convert to 0-based index
        else:
            start_idx = 0

        if stop_row is not None:
            end_idx = stop_row  # stop_row is inclusive, so no -1
        else:
            end_idx = len(df)

        # Validate range
        start_idx = max(0, start_idx)
        end_idx = min(len(df), end_idx)

        if start_idx >= end_idx:
            print(f"Error: Invalid row range. Start row must be less than end row.")
            return

        # Filter dataframe to selected range
        df_subset = df.iloc[start_idx:end_idx].copy()
        print(f"Processing rows {start_idx + 1} to {end_idx} ({len(df_subset)} rows) from {input_file}...")

        # Create output with only columns B and C
        import os
        script_name = os.path.splitext(os.path.basename(__file__))[0]


        # Prepare data for multiprocessing (only selected rows)
        row_data = [(index, row['text'], row['ml']) for index, row in df_subset.iterrows()]

        # Use multiprocessing to process rows in parallel
        if num_processes is None:
            num_processes = mp.cpu_count()

        print(f"Using {num_processes} CPU cores for parallel processing...")

        with mp.Pool(processes=num_processes) as pool:
            # Use imap with progress bar
            results = []
            with tqdm(total=len(row_data), desc=f"Processing rows {start_idx + 1}-{end_idx}") as pbar:
                for result in pool.imap(process_row, row_data):
                    results.append(result)
                    pbar.update(1)

        # Update DataFrame with results
        print("Updating results...")
        for index, json_content, icd_codes in results:
            df.at[index, f'{script_name}_description'] = json_content
            df.at[index,f'{script_name}'] = icd_codes


        # Save the extracted results
        df.to_csv(output_file, index=False)
        print(f"\nCompleted! Results saved to {output_file}")
    

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except KeyError:
        print("Error: Required columns 'text' or 'ml' not found in CSV file.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict ICD-10 codes for discharge summary notes using multicore processing')
    parser.add_argument('-i', '--input', default='Claude.csv', help='Input CSV file (default: Claude.csv)')
    parser.add_argument('-o', '--output', default='Claude.csv', help='Output CSV file (default: Claude.csv)')
    parser.add_argument('-p', '--processes', type=int, help='Number of processes to use (default: all CPU cores)')
    parser.add_argument('-s', '--start', type=int, help='Start row number (1-based, default: 1)')
    parser.add_argument('-e', '--end', type=int, help='End row number (1-based, default: last row)')

    args = parser.parse_args()
    main(args.input, args.output, args.processes, args.start, args.end)