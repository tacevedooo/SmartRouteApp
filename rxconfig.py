import reflex as rx
import os

port = int(os.environ.get("PORT", 7860))

config = rx.Config(
    app_name="SmartRouteApp",
    api_url=f"http://localhost:{port}",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin()
    ]
)