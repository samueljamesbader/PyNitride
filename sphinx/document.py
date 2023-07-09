import subprocess
subprocess.run(["sphinx-apidoc","-o", "auto", "../", "../setup.py",  "-f", "-e"])
subprocess.run(["sphinx-build","-b","html",".","../docs/html"])
subprocess.run(["cp","index_redirect.html","../docs/html/index.html"])
subprocess.run(["cp","../docs/html/.nojekyll","../docs/.nojekyll"])
