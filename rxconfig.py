import reflex as rx

config = rx.Config(
    app_name="SmartRouteApp",
    api_url="smartrouteapp-production.up.railway.app",
    deploy_url="smartrouteapp-production.up.railway.app",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin()
    ]
)