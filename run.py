import os

os.environ['NEWSAPI_KEY'] = open("keys.txt").read().strip()

if __name__ == "__main__":
    os.system("streamlit run app.py")
