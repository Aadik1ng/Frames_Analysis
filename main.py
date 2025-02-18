from flask import Flask
from routes.routes import routes_blueprint
from init.initialize import initialize
import os
app = Flask(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
# Run one-time initialization tasks (e.g., creating DB tables)
initialize()

# Register API routes under the /api prefix
app.register_blueprint(routes_blueprint, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True)
