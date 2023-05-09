from flask import Flask
from flask.typing import RouteCallable


class FlaskApi(object):

    def __init__(self, **configs):
        self.app = Flask(__name__)
        self.configs(**configs)

    def configs(self, **configs):
        for config, value in configs:
            self.app.config[config.upper()] = value

    def add_endpoint(self,
                     endpoint: str = "",
                     endpoint_name: str | None = None,
                     handler: RouteCallable | None = None,
                     methods: list = ["GET"],
                     *args,
                     **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name,
                              handler, methods=methods, *args, **kwargs)

    def run(self, **kwargs):
        self.app.run(**kwargs)
