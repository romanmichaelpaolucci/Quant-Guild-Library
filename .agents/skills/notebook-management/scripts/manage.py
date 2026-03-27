import os
import argparse
import sys

def get_notebooks(root_dir):
    notebooks = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.ipynb'):
                notebooks.append(os.path.join(root, file))
    return notebooks

def action_status(root_dir):
    print(f"{'Notebook Path':<60} | {'PY':<3} | {'MD':<3}")
    print("-" * 75)
    notebooks = get_notebooks(root_dir)
    for nb in notebooks:
        py_exists = os.path.exists(nb.replace('.ipynb', '.py'))
        md_exists = os.path.exists(nb.replace('.ipynb', '.md'))
        rel_path = os.path.relpath(nb, root_dir)
        print(f"{rel_path:<60} | {'[x]' if py_exists else '[ ]'} | {'[x]' if md_exists else '[ ]'}")

def action_clean(root_dir):
    notebooks = get_notebooks(root_dir)
    for nb in notebooks:
        py_path = nb.replace('.ipynb', '.py')
        md_path = nb.replace('.ipynb', '.md')
        
        if os.path.exists(py_path):
            print(f"Removing {py_path}...")
            os.remove(py_path)
            
        if os.path.exists(md_path):
            print(f"Removing {md_path}...")
            os.remove(md_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Jupyter Notebook conversions.")
    parser.add_argument("--action", choices=['status', 'clean'], required=True, help="Action to perform.")
    parser.add_argument("path", help="Directory path to scan.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a valid directory.")
        sys.exit(1)
        
    if args.action == 'status':
        action_status(args.path)
    elif args.action == 'clean':
        action_clean(args.path)
