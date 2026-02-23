"""
Compatibility shim for the old `serving` import path.

This repo is migrating to the canonical namespace:
`ml_lifecycle_platform.serving`.

Remove this shim once all references are updated.
"""

from ml_lifecycle_platform.serving import *  # noqa: F401,F403
