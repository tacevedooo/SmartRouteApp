import reflex as rx

config = rx.Config(
    app_name="SmartRouteApp",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin()
    ]
)