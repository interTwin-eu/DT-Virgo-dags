from app.main.main import main_bp


@main_bp.app_errorhandler(500)
def handle_bad_request(err):
    return 'Internal error!', 500