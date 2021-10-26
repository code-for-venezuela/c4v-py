import sys
from logger import logger

mkdocs_build_output = sys.argv[1]
if mkdocs_build_output != str(0):
    logger.error('Error building mkdocs. Warnings were found.')
    raise Exception('Error building mkdocs. Warnings were found.')
else:
    logger.info('No warnings found building mkdocs.')
    
