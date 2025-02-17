from flask import Flask
from routes.routes import routes_blueprint
from init.initialize import initialize

app = Flask(__name__)

# Run one-time initialization tasks (e.g., creating DB tables)
initialize()

# Register API routes under the /api prefix
app.register_blueprint(routes_blueprint, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True)
