import reflex as rx

config = rx.Config(
    app_name="SmartRouteApp",
    frontend_port=3000,
    backend_port=8000,
    api_url="http://smartrouteapp.railway.internal/",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin()
    ]
)