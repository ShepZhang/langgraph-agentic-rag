"""Run the Gradio application."""

from __future__ import annotations

from config import get_settings
from ui.gradio_app import create_app


def main() -> None:
    """Launch the Gradio UI."""

    settings = get_settings()
    app = create_app()
    app.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )


if __name__ == "__main__":
    main()
