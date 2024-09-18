import os


def list_non_hidden_files(start_path):
    """
    List all non-hidden files starting from the given path, showing paths relative to start_path.

    Args:
    start_path (str): The directory to start the search from.

    Returns:
    None: Prints relative file paths to console.
    """
    for root, dirs, files in os.walk(start_path):
        # Remove hidden directories from the list
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            # Skip hidden files
            if not file.startswith('.'):
                # Get the full path
                full_path = os.path.join(root, file)
                # Get the relative path
                relative_path = os.path.relpath(full_path, start_path)
                print(relative_path)


# Get the current working directory
current_dir = os.getcwd()

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# List all non-hidden files starting from the parent directory
print(f"Listing all non-hidden files relative to: {os.path.basename(parent_dir)}")
list_non_hidden_files(parent_dir)