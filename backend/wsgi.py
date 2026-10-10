from backend.app import create_app
app = create_app()
if __name__ == "__main__":
    import os
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
