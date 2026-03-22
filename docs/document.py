import subprocess
import shutil
import pathlib
import argparse
import webbrowser

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true",
        help="Open the generated docs in a browser after building")
args = parser.parse_args()

# Paths
sphinx_dir = pathlib.Path(__file__).parent
root = sphinx_dir.parent

# Generate the .rst files for the API reference
subprocess.run(["sphinx-apidoc", "-o", str(sphinx_dir/"auto"),
                str(root/"src"/"pynitride"), "-f", "-e"],check=True)

# Build the HTML documentation
subprocess.run(["sphinx-build", "-b", "html", str(sphinx_dir),
                str(sphinx_dir / "build" / "html")],check=True)

# Set up a redirect from general index.html to the manual index page
shutil.copy(sphinx_dir / "index_redirect.html",
            sphinx_dir / "build" / "html" / "index.html")

# Copy the .nojekyll file (because we serve from build/ not build/html/)
shutil.copy(sphinx_dir / "build" / "html" / ".nojekyll",
            sphinx_dir / "build" / ".nojekyll")

# Open the generated docs in a browser if --show was specified
if args.show:
    webbrowser.open((sphinx_dir / "build" / "html" / "index.html").as_uri())

