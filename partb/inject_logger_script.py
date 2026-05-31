import re
import glob
import os

def inject(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'from partb.logger import' in content:
        return
        
    lines = content.split('\n')
    out = []
    has_injected = False
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('def '):
            if i > 0 and lines[i-1].strip().startswith('@'):
                # insert before the existing decorator
                pass # actually it's easier to just put it after existing decorators
            indent = line[:len(line) - len(stripped)]
            out.append(f'{indent}@time_it')
            out.append(line)
            has_injected = True
        elif stripped.startswith('async def '):
            indent = line[:len(line) - len(stripped)]
            out.append(f'{indent}@async_time_it')
            out.append(line)
            has_injected = True
        else:
            out.append(line)
            
    if has_injected:
        # add imports
        import_str = "from partb.logger import time_it, async_time_it\n"
        idx = 0
        for j, l in enumerate(out):
            if l.startswith('import ') or l.startswith('from '):
                idx = j
        out.insert(idx, import_str)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out))
        print(f"Injected into {filepath}")

for root, dirs, files in os.walk('.'):
    if '.venv' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py') and file not in ['logger.py', 'test_connections.py', 'download_models.py']:
            inject(os.path.join(root, file))
