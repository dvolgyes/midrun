"""Enable check for one hook script.

The execute bit is the only enable switch, re-checked immediately before every use
(the current stat is never trusted from an earlier check or cache): `chmod +x`/`-x`
take effect at the next applicable callback, including within the same epoch.
"""

import os
import stat as stat_module
from pathlib import Path


def resolve_enabled_script(live_dir: Path, modality: str, name: str) -> Path | None:
    """Returns `live_dir/modality/name.sh` if it is currently enabled, else `None`.

    "Enabled" means a regular file, not a symlink, with at least one POSIX execute
    bit set. Callers must call this again immediately before launch -- the exec bit,
    or the file itself, can change at any moment between checks.
    """
    # This runs at every callback, mostly for disabled scripts, so the path stays a
    # string until the script is known to be enabled: a pathlib join costs more
    # than the lstat itself.
    script_path = f"{live_dir}{os.sep}{modality}{os.sep}{name}.sh"
    try:
        st = os.lstat(script_path)
    except OSError:
        return None
    if stat_module.S_ISLNK(st.st_mode) or not stat_module.S_ISREG(st.st_mode):
        return None
    if not st.st_mode & 0o111:
        return None
    return Path(script_path)
