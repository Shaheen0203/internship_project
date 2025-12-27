import sys

# Add your project directory to the sys.path
path = '/home/yourusername/mysite'
if path not in sys.path:
    sys.path.insert(0, path)

# Import the Flask app
from app import app as application

# No need to call app.run() - PythonAnywhere's server will handle this