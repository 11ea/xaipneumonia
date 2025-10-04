from django.utils.deprecation import MiddlewareMixin

class COOPCOEPMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        response["Cross-Origin-Embedder-Policy"] = "require-corp"
        response["Cross-Origin-Opener-Policy"] = "same-origin"
        return response
