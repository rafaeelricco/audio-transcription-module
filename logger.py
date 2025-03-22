class Logger:
    VERBOSE = False

    @staticmethod
    def log(sucesso: bool, mensagem: str, nivel: str = "info"):
        if nivel == "debug" and not Logger.VERBOSE:
            return

        prefixo = {
            "info": "",
            "debug": "[DEBUG]",
            "warning": "[WARNING]",
            "error": "[ERROR]",
        }.get(nivel, "")

        print(f"{'✓' if sucesso else '✗'} {prefixo} {mensagem}")

    @classmethod
    def set_verbose(cls, verbose: bool):
        cls.VERBOSE = verbose
