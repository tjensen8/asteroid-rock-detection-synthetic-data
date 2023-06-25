"""
File runs each of the preprocess scripts and skip to the next once complete or if there is an error.
"""

import preprocess_artificial_mask
import preprocess_artificial_render

try:
    preprocess_artificial_mask
except Exception as e:
    print(e)
    next

try:
    preprocess_artificial_render
except Exception as e:
    print(e)
    next