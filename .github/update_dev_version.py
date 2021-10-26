from packaging import version
import time
import sys
import fileinput
from datetime import datetime

"""
    This script modifies a python PEP404 version by 
    suffixing a development release segment string.
    
    Finally it will save the new version to the 
    project.toml file
    
    Args:
        -version: the current version of the project. 
"""

DATE_FORMAT = "%Y%m%d%H%M"


def concat_dev_segment(version_str: str) -> str:
    v = version.parse(version_str)
    return v.base_version + ".dev" + datetime.today().strftime(DATE_FORMAT)


current_version = sys.argv[1]
dev_version = concat_dev_segment(current_version)
for line in fileinput.input("pyproject.toml", inplace=True):
    if line.startswith('version = "{}"'.format(current_version)):
        line = line.replace('version = "{}"'.format(current_version),
                            'version = "{}"'.format(dev_version))
    sys.stdout.write(line)
