import pdoc
from pathlib import Path
import os
import sys
files = list(Path("./src/").rglob("*.py"))


for file in files:
    path = os.path.basename(file).split('.')
    if path[0] !='__init__':
        module_path = os.path.abspath(file)
        sys.path.append(module_path)
        print(module_path)
        pdoc.import_module(module_path)
        # h = pdoc.html(module_path, source=False)
