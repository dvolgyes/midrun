"""The HTML page of one virtual directory, as `GET /files/<dir>` shows it to a browser."""

from html import escape
from urllib.parse import quote


def virtual_dir_page(path: str, entries: list[tuple[str, bool]]) -> str:
    """Renders `path`'s entries as links under `/files`.

    Each name is escaped for the HTML text and percent-encoded for the link, so a
    name cannot add markup.

    Args:
        path: The directory's virtual path, such as `/` or `/model`.
        entries: Each child's name and whether it is a directory.

    Returns:
        A complete HTML document without scripts or styles.
    """
    base = "/files" + quote(path.rstrip("/"))
    items = []
    if path != "/":
        parent = quote(path.rstrip("/").rsplit("/", 1)[0])
        items.append(f'<li><a href="/files{escape(parent)}/">../</a></li>')
    for name, is_dir in entries:
        href = escape(f"{base}/{quote(name, safe='')}")
        items.append(
            f'<li><a href="{href}">{escape(name)}{"/" if is_dir else ""}</a></li>'
        )
    title = escape(path)
    return (
        '<!doctype html>\n<html lang="en">\n<head><meta charset="utf-8">'
        f"<title>midrun files: {title}</title></head>\n"
        f"<body>\n<h1>{title}</h1>\n<ul>\n"
        + "\n".join(items)
        + "\n</ul>\n</body>\n</html>\n"
    )
