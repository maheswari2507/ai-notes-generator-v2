from flask import Flask, render_template, request
import os
from utils.pdf_generator import generate_pdf


from utils.pdf_extractor import extract_text
from utils.transformer_summarizer import generate_summary as ai_summary

print("App is starting...")

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/paste')
def paste():
    return render_template('paste.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['user_text']
    if not text.strip():
        return "No text provided."
    result = ai_summary(text)
    return render_template(
        'result.html',
        overall_summary=result["overall_summary"],
        paragraphs=result["paragraphs"],
        key_points=result["key_points"],
    )


@app.route('/upload', methods=['POST'])
def upload():
    print("STEP 1: Upload route entered")
    file     = request.files['pdf_file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print("STEP 2: File saved →", filepath)

    text = extract_text(filepath)
    print("STEP 3: Text extracted —", len(text.split()), "words")

    result = ai_summary(text)
    print("STEP 4: Done")

    return render_template(
        'result.html',
        overall_summary=result["overall_summary"],
        paragraphs=result["paragraphs"],
        key_points=result["key_points"],
    )
@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    overall_summary = request.form.get('overall_summary', '')
    paragraphs      = request.form.getlist('paragraphs')
    key_points      = request.form.getlist('key_points')
    return generate_pdf(overall_summary, paragraphs, key_points)


if __name__ == '__main__':
    app.run(debug=True)