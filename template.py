import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

files_list = [
    'src/__init__.py',
    'src/helper.py',
    'src/prompt.py',
    '.env',
    'setup.py',
    'app.py',
    'research/test.ipynb'
]

for file in files_list:
    file_path = Path(file)
    filedir, filename = os.path.split(file_path)

    if filedir !='':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating "{filedir}" directory for "{filename}" file...')

    if not(os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
        with open(file_path, 'w') as f:
            pass
            logging.info(f'Creating "{file_path}" empty file...')
    else:
        logging.info(f'{file_path} already exists!')

    
    