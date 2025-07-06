from flask import Flask, render_template, request
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, re, html, emoji, os
from bs4 import BeautifulSoup
from langdetect import detect
from collections import Counter
from io import BytesIO
import base64
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
app = Flask(__name__)

os.makedirs("feedback", exist_ok=True)

# Load model and tokenizer
MODEL_NAME = "ojasvagupta/YT-senti-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# YouTube API
API_KEY = "AIzaSyAcIFihKE87xfCz3WSlJx_Lut_PwJBn0hI"
youtube = build("youtube", "v3", developerKey=API_KEY)

# Text cleaning
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)

    # ✅ Convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))  # So :smile: becomes ' smile '

    # ✅ Remove URLs
    text = re.sub(r"http\S+|www\S+", '', text)

    # ✅ Remove other unwanted characters
    text = re.sub(r"[^A-Za-z0-9\s:_]", '', text)  # Allow underscores and colons in demojized names
    return text.lower().strip()


# Emoji extraction
def extract_emojis(text):
    return [char for char in text if emoji.is_emoji(char)]

# Fetch YouTube comments
def get_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id,
        maxResults=100, textFormat="plainText"
    )
    response = request.execute()
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    return comments

# Analyze with model
def analyze_comments(comments):
    results, emojis = [], []
    for comment in comments:
        cleaned = clean_text(comment)
        em = extract_emojis(comment)
        if not cleaned:
            continue
        try:
            if detect(cleaned) != "en":
                continue
        except:
            continue

        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**inputs)
            pred = torch.argmax(output.logits, dim=1).item()
            sentiment = labels[pred]
            results.append({
                "Original": comment,
                "Cleaned": cleaned,
                "Sentiment": sentiment
            })
        emojis.extend(em)
    return results, emojis

# Chart utils
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def generate_pie_chart(results):
    count = Counter(r["Sentiment"] for r in results)
    fig, ax = plt.subplots()
    ax.pie(count.values(), labels=count.keys(), autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    return fig_to_base64(fig)

def generate_emoji_chart(emojis):
    from matplotlib import font_manager

    top_emojis = Counter(emojis).most_common(10)
    if not top_emojis:
        return None

    emoji_chars, freqs = zip(*top_emojis)
    fig, ax = plt.subplots()

    bars = ax.bar(range(len(emoji_chars)), freqs, color="#ffcc00")

    # ✅ Set proper font for emojis
    try:
        emoji_font = None
        for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
            if any(x in font.lower() for x in ['seguiemj', 'apple color emoji', 'noto color emoji']):
                emoji_font = font
                break

        if emoji_font:
            font_properties = font_manager.FontProperties(fname=emoji_font)
            ax.set_xticks(range(len(emoji_chars)))
            ax.set_xticklabels(emoji_chars, fontproperties=font_properties, fontsize=16)
        else:
            ax.set_xticks(range(len(emoji_chars)))
            ax.set_xticklabels(emoji_chars, fontsize=16)
            print("⚠️ Emoji font not found, using default.")
    except Exception as e:
        ax.set_xticks(range(len(emoji_chars)))
        ax.set_xticklabels(emoji_chars, fontsize=16)
        print(f"Emoji font load error: {e}")

    ax.set_title("Top Emojis Used")
    return fig_to_base64(fig)


# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_url = request.form["video_url"]
        video_id = video_url.split("v=")[-1].split("&")[0]
        comments = get_comments(video_id)
        results, emoji_list = analyze_comments(comments)

        pie = generate_pie_chart(results)
        emoji_chart = generate_emoji_chart(emoji_list)

        return render_template("results.html", results=results, pie_chart=pie,
                               emoji_chart=emoji_chart, video_url=video_url)
    return render_template("index.html")

@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    feedback_data = []

    for key in request.form:
        if key.startswith("feedback_"):
            idx = int(key.split("_")[1])
            user_feedback = request.form.get(key)

            if user_feedback:  # ✅ Only save if user gave a label
                feedback_data.append({
                    "Original": request.form.get(f"original_{idx}"),
                    "Cleaned": request.form.get(f"cleaned_{idx}"),
                    "Predicted": request.form.get(f"predicted_{idx}"),
                    "UserFeedback": user_feedback
                })

    # ✅ Save only if some feedback was actually provided
    if feedback_data:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        pd.DataFrame(feedback_data).to_csv(f"feedback/user_feedback_{ts}.csv", index=False)

    return render_template("thankyou.html")


from flask import Response

@app.route("/download", methods=["POST"])
def download_csv():
    from io import StringIO
    results = []
    count = int(request.form["count"])

    for i in range(count):
        results.append({
            "Original": request.form.get(f"original_{i}"),
            "Cleaned": request.form.get(f"cleaned_{i}"),
            "Sentiment": request.form.get(f"predicted_{i}")
        })

    df = pd.DataFrame(results)
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=analysis_result.csv"}
    )


    
if __name__ == "__main__":
    app.run(debug=True)
