from flask import Flask

def create_app():
    dash = Flask(__name__)
    
    # Configuration de l'application (si nécessaire)
    dash.config.from_object('config.Config')
    
    # Importation des routes
    from . import routes
    dash.register_blueprint(routes.bp)

    return dash
