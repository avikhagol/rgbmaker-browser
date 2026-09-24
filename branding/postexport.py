#!/usr/bin/env python3
"""Brand a `marimo export html-wasm` output folder (no dependencies).

    python3 branding/postexport.py _site      # CI
    python3 branding/postexport.py docs       # local export

marimo always writes its own manifest ("A Marimo App") and icons, and app.py
cannot change them, so this runs after every export.
"""
import json
import pathlib
import shutil
import sys

NAME = "RGBMaker"
SHORT_NAME = "RGBMaker"
DESCRIPTION = "Radio-optical composite images (ROR, IOU, optical RGB with contours) in the browser"
THEME = "#000000"
BACKGROUND = "#ffffff"

ICONS = [
    {"src": "android-chrome-192x192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
    {"src": "android-chrome-512x512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
    {"src": "maskable-512x512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
]


def main(out: pathlib.Path) -> None:
    if not (out / "index.html").exists():
        sys.exit(f"{out}/index.html not found: run marimo export html-wasm first")
    for icon in (pathlib.Path(__file__).parent / "icons").iterdir():
        shutil.copy2(icon, out / icon.name)

    manifest = {
        "name": NAME,
        "short_name": SHORT_NAME,
        "description": DESCRIPTION,
        "icons": ICONS,
        "start_url": ".",
        "scope": ".",
        "display": "standalone",
        "theme_color": THEME,
        "background_color": BACKGROUND,
    }
    # relative paths so it works under the /rgbmaker-browser/ project path
    for name in ("manifest.json", "site.webmanifest"):
        (out / name).write_text(json.dumps(manifest, indent=2) + "\n")

    index = out / "index.html"
    html = index.read_text()
    html = html.replace('content="a marimo app"', f'content="{DESCRIPTION}"')
    if 'href="./favicon-32x32.png"' not in html:
        html = html.replace(
            '<link rel="icon" href="./favicon.ico" />',
            '<link rel="icon" href="./favicon.ico" sizes="48x48" />\n'
            '    <link rel="icon" type="image/png" sizes="32x32" href="./favicon-32x32.png" />\n'
            '    <link rel="icon" type="image/png" sizes="16x16" href="./favicon-16x16.png" />',
        )
    index.write_text(html)
    print(f"branded {out}: name={NAME!r}, icons={len(list((pathlib.Path(__file__).parent / 'icons').iterdir()))}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
