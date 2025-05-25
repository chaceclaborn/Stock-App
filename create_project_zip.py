import os
import zipfile
from datetime import datetime
from pathlib import Path

def create_project_zip():
    """
    Creates a zip file of the stock project, excluding specified files and folders.
    """
    
    # Define exclusions - BE VERY SPECIFIC
    EXCLUDE_FOLDERS = {
        'data_cache',
        'venv',
        '.venv',  # Your actual venv folder
        '__pycache__',
        '.git',
        '.idea',
        '.vscode',
        'node_modules',
    }
    
    EXCLUDE_FILES = {
        'stock_data.db',
        'stock_generate_structure.py',
        '.DS_Store',
        'Thumbs.db',
    }
    
    EXCLUDE_EXTENSIONS = {
        '.pyc',
        '.pyo',
        '.pyd',
    }
    
    # Get the current directory
    project_root = Path.cwd()
    
    # Fixed filename (will overwrite if exists)
    zip_filename = "stock_app_clean.zip"
    
    # Remove existing zip if it exists
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        print(f"Removed existing {zip_filename}")
    
    included_files = []
    excluded_count = 0
    
    print(f"Creating zip file: {zip_filename}")
    print(f"Excluding folders: {EXCLUDE_FOLDERS}")
    print("-" * 50)
    
    # Walk through all files
    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)
        
        # Get relative path for checking
        try:
            rel_path = root_path.relative_to(project_root)
            rel_parts = rel_path.parts
        except ValueError:
            continue
        
        # Check if we're inside an excluded folder
        skip_dir = False
        for part in rel_parts:
            if part in EXCLUDE_FOLDERS:
                skip_dir = True
                break
        
        if skip_dir:
            # Clear dirs to prevent walking into subdirectories
            dirs.clear()
            continue
        
        # Also check the immediate subdirectories and remove excluded ones
        dirs[:] = [d for d in dirs if d not in EXCLUDE_FOLDERS]
        
        # Process files in current directory
        for file in files:
            file_path = root_path / file
            
            # Skip the zip file itself
            if file == zip_filename:
                continue
            
            # Skip excluded files
            if file in EXCLUDE_FILES:
                excluded_count += 1
                print(f"❌ Excluding: {file}")
                continue
            
            # Skip by extension
            if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                excluded_count += 1
                continue
            
            # Skip this script
            if file == 'create_project_zip.py':
                continue
            
            # Get relative path for the zip
            arcname = file_path.relative_to(project_root)
            
            # Final check - make sure we're not including anything from excluded folders
            arcname_str = str(arcname).replace('\\', '/')
            if any(f"/{folder}/" in f"/{arcname_str}" or arcname_str.startswith(f"{folder}/") for folder in EXCLUDE_FOLDERS):
                excluded_count += 1
                continue
            
            included_files.append((file_path, arcname))
    
    # Create the zip with collected files
    print(f"\nAdding {len(included_files)} files to zip...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, arcname in included_files:
            zipf.write(file_path, arcname)
            print(f"✓ {arcname}")
    
    print("-" * 50)
    print(f"✅ Zip file created: {zip_filename}")
    print(f"📁 Files included: {len(included_files)}")
    print(f"❌ Files excluded: {excluded_count}")
    
    # Get zip file size
    zip_size = os.path.getsize(zip_filename)
    print(f"💾 Zip file size: {zip_size / (1024 * 1024):.2f} MB")
    
    return zip_filename

def verify_no_venv(zip_filename):
    """
    Double-check that no venv files made it in
    """
    print("\n🔍 Verifying no venv files...")
    
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        contents = zipf.namelist()
        venv_files = [f for f in contents if '.venv' in f or 'venv/' in f or 'venv\\' in f]
        
        if venv_files:
            print(f"❌ ERROR: Found {len(venv_files)} venv files in zip!")
            for f in venv_files[:5]:
                print(f"  - {f}")
            if len(venv_files) > 5:
                print(f"  ... and {len(venv_files) - 5} more")
            return False
        else:
            print("✅ Verified: No venv files in zip!")
            return True

def list_zip_structure(zip_filename):
    """
    Show the top-level structure of the zip
    """
    print("\n📂 Zip contents (top level):")
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        contents = zipf.namelist()
        
        # Get unique top-level items
        top_level = set()
        for item in contents:
            parts = item.split('/')
            if len(parts) == 1:
                top_level.add(item)
            else:
                top_level.add(parts[0] + '/')
        
        for item in sorted(top_level):
            print(f"  {item}")

if __name__ == "__main__":
    try:
        # Create the zip file
        zip_file = create_project_zip()
        
        # Verify no venv files
        success = verify_no_venv(zip_file)
        
        if success:
            # Show structure
            list_zip_structure(zip_file)
        else:
            print("\n⚠️  WARNING: The zip contains venv files! Please check the exclusion logic.")
            
    except Exception as e:
        print(f"❌ Error creating zip file: {e}")
        import traceback
        traceback.print_exc()