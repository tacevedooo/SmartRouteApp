import reflex as rx

config = rx.Config(
    app_name="SmartRouteApp",
    api_url="https://TU-DOMINIO.up.railway.app",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin()
    ]
)