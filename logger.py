class Logger:
    @staticmethod
    def log(sucesso: bool, mensagem: str):
        print(f"{'✓' if sucesso else '✗'} {mensagem}")
