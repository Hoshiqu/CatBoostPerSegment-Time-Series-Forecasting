import sys
from streamlit.web import cli
from streamlit.web.cli import configurator_options, main
from pathlib import Path
import os
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def get_available_port(start_port, max_attempts):
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return None

@main.command("runapp")
@configurator_options
def run_streamlit_app(**kwargs):
    """Runs the Streamlit application."""
    cli.bootstrap.load_config_options(flag_options=kwargs)
    app_directory = Path(__file__).parent
    filename = str(app_directory.joinpath("run.py"))  # .abspath()

    # Set the desired starting port and maximum attempts to find an available port
    start_port = 8051
    max_attempts = 10

    # Find an available port to use
    port = get_available_port(start_port, max_attempts)

    if port is None:
        print("Unable to find an available port to run the application.")
        return

    # Update the Streamlit server port
    kwargs["--server.port"] = str(port)

    # Run the Streamlit application
    cli._main_run(filename, flag_options=kwargs)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    kwargs = {
        "--server.enableXsrfProtection=false",
        "--server.enableCORS=true",
        "--global.developmentMode=false",
    }
    sys.exit(run_streamlit_app(kwargs))
