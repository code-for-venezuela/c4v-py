'''
    This script provides functionality for staging tags in documents.
    It will remove documents sections enclosed with "##STAGING##" tags (without quotes).  
    This is intended to run before the building process of the documentation.
'''

import os
from typing import List
from logger import logger

# Only process files and folders within
FOLDERS = ('docs_en', 'docs_es')
ACTIVE_FILE_EXT = ('.md')
START_PATH = './docs/docs/'
STAGE_TAG = '## stage ##\n'

def get_stage_tags_positions(list_of_elems, element):
    '''Returns the indexes of all occurrences of give element in
    the list- listOfElements'''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

def sanitize_text(stage_tag_positions: List[int], lines: List[str]):
    '''Returns the text lines that are not enclosed in ## stage ## tags'''
    stage_tags_pairs = zip(stage_tag_positions[::2], stage_tag_positions[1::2])
    c = 0
    for p1, p2 in stage_tags_pairs:
        del lines[p1:p2+1]
        c += 1
    logger.info(f'Total sanitized entries {c}')
    return lines    

def sanitize_file(root, file_name: str):
    file = os.path.join(root, file_name)
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    stage_tags_linear = get_stage_tags_positions(lines, STAGE_TAG)
    logger.info(f'Sanitizing file {file}...')
    if len(stage_tags_linear) == 0:
        logger.info('No staging sections to remove')
        return    
    newText = sanitize_text(stage_tags_linear, lines)
    f = open(file, 'w')
    f.writelines(newText)
    f.close()
    
def is_markdown_file(filename):
    return filename.endswith(ACTIVE_FILE_EXT)

def process_files(top_tuple):
    for topfolder, subfolder, filesintop in top_tuple:
        """Process files in topfolder first"""
        for file in filesintop:
            if is_markdown_file(file):
                sanitize_file(topfolder, file)
        """Process subfolders"""
        for folder in subfolder:
            folder_walk = os.walk(os.path.join(topfolder, folder))
            process_files(folder_walk)

root_folders_walk = [ walk for walk in os.walk(START_PATH) if walk[0].endswith(FOLDERS) ]
process_files(root_folders_walk)
