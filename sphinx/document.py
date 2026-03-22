import subprocess
import shutil
import pathlib
import argparse
import webbrowser

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="Open the generated docs in a browser after building")
args = parser.parse_args()

sphinx_dir = pathlib.Path(__file__).parent
root = sphinx_dir.parent

subprocess.run(["sphinx-apidoc", "-o", str(sphinx_dir/"auto"), str(root/"src"/"pynitride"), "-f", "-e"],check=True)
subprocess.run(["sphinx-build", "-b", "html", str(sphinx_dir), str(root / "docs" / "html")],check=True)
shutil.copy(sphinx_dir / "index_redirect.html", root / "docs" / "html" / "index.html")
shutil.copy(root / "docs" / "html" / ".nojekyll", root / "docs" / ".nojekyll")

if args.show:
    webbrowser.open((root / "docs" / "html" / "index.html").as_uri())

