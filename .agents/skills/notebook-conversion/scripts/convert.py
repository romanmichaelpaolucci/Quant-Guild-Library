import os
import nbformat
import sys
import argparse

def convert_notebook(ipynb_path, output_dir=None):
    """Converts a single .ipynb file to .md using nbconvert and also generates a .py script."""
    if not ipynb_path.endswith('.ipynb'):
        print(f"Skipping {ipynb_path}: Not a .ipynb file.")
        return

    print(f"Converting {ipynb_path}...")
    try:
        # 1. Convert to Markdown using nbconvert
        cmd_md = [
            'jupyter', 'nbconvert', '--to', 'markdown', ipynb_path
        ]
        if output_dir:
            cmd_md.extend(['--output-dir', output_dir])
        subprocess.run(cmd_md, check=True)

        # 2. Convert to Python script using nbconvert (more standard)
        cmd_py = [
            'jupyter', 'nbconvert', '--to', 'python', ipynb_path
        ]
        if output_dir:
            cmd_py.extend(['--output-dir', output_dir])
        subprocess.run(cmd_py, check=True)
            
    except Exception as e:
        print(f"Error converting {ipynb_path}: {e}")

def convert_recursive(root_dir, output_dir=None):
    """Recursively finds and converts all .ipynb files in a directory."""
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories like .git and .venv
        dirs[:] = [d for d in dirs if not d.startswith('.')]
            
        for file in files:
            if file.endswith('.ipynb'):
                ipynb_path = os.path.join(root, file)
                convert_notebook(ipynb_path, output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Jupyter Notebooks to .py and .md using nbconvert.")
    parser.add_argument("path", help="Path to a .ipynb file or a directory.")
    parser.add_argument("--output-dir", help="Optional output directory for converted files.")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        convert_recursive(args.path, output_dir=args.output_dir)
    elif os.path.isfile(args.path):
        convert_notebook(args.path, output_dir=args.output_dir)
    else:
        print(f"Error: {args.path} is not a valid file or directory.")
        sys.exit(1)
