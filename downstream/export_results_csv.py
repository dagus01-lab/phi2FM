import os
import json
import pandas as pd

# Root directory where model result folders are stored
ROOT_DIR = "/Data/phi2FM_models/finetuning/" #"/Data/phi2FM_models/finetuning/"

# Function to collect JSON result files per downstream task
def collect_results(root_dir):
    """
    Traverse the folder structure and collect result JSONs per downstream task.
    Expected structure: {model_name}/{downstream_task}/{downstream_task}/{final_dir}/results.json
    """
    task_data = {}
    for model_name in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        # Each model has downstream_task subdirectories
        for task in os.listdir(model_path):
            task_path = os.path.join(model_path, task)
            # Some structures repeat the task name twice
            nested = os.path.join(task_path, task)
            search_path = nested if os.path.isdir(nested) else task_path
            # Iterate over final directories
            for final_dir in os.listdir(search_path):
                dir_path = os.path.join(search_path, final_dir)
                if not os.path.isdir(dir_path):
                    continue
                # Find JSON files inside dir_path
                for fname in os.listdir(dir_path):
                    if fname.endswith('.json'):
                        file_path = os.path.join(dir_path, fname)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        # Initialize list for this task
                        task_data.setdefault(task, []).append({
                            'model': model_name,
                            'setting': final_dir,
                            **flatten_dict(data)
                        })
    return task_data


def flatten_dict(d, parent_key='', sep='__'):
    """
    Flatten nested dictionaries: { 'a': {'b': 1} } -> {'a__b': 1}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def export_to_excel(task_data, output_dir="./excel_outputs"):
    """
    For each downstream task, export the collected data to an Excel file with one sheet.
    """
    os.makedirs(output_dir, exist_ok=True)
    for task, records in task_data.items():
        df = pd.DataFrame(records)
        # Define output file per task
        output_file = os.path.join(output_dir, f"{task}_results.xlsx")
        # Write to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=task)
        print(f"Exported {len(records)} records for task '{task}' to {output_file}")


if __name__ == "__main__":
    task_results = collect_results(ROOT_DIR)
    export_to_excel(task_results)
