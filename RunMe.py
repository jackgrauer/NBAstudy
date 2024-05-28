import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

# Add the EDA and MAIN directories to sys.path
eda_dir = os.path.join(root_dir, 'EDA')
main_dir = os.path.join(root_dir, 'MAIN')
sys.path.append(eda_dir)
sys.path.append(main_dir)

# Import common module from MAIN directory
try:
    import MAIN.common as common
    print("common module imported successfully")
except ImportError as e:
    print(f"Error importing common: {e}")

# Import data.raw modules
try:
    from data.raw import EDA as raw_EDA
    from data.raw import MAIN as raw_MAIN
    print("raw_EDA and raw_MAIN modules imported successfully")
except ImportError as e:
    print(f"Error importing raw modules: {e}")

# Import EDA and MAIN processing modules
try:
    import EDA.EDA as eda_processor
    import MAIN.MAIN as main_processor
    print("EDA and MAIN processing modules imported successfully")
except ImportError as e:
    print(f"Error importing processing modules: {e}")

# Process EDA and MAIN
try:
    eda_processor.process(raw_EDA)
    main_processor.process(raw_MAIN)
    print("Processing completed successfully")
except NameError as e:
    print(f"Error during processing: {e}")
except Exception as e:
    print(f"Unexpected error during processing: {e}")