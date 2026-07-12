import os
from flask import Flask, render_template, request
from utils.pdf_generator import generate_pdf
from utils.pdf_extractor import extract_text
from utils.transformer_summarizer import generate_summary as ai_summary

print("App is starting...")

app = Flask(__name__)

# ─────────────────────────────────────────────
#  Upload folder setup
# ─────────────────────────────────────────────

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route('/')
def home():
    """Render the main landing page."""
    return render_template('index.html')


@app.route('/paste')
def paste():
    """Render the paste-text input page."""
    return render_template('paste.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle pasted text input and return summarized result."""
    text = request.form.get('user_text', '').strip()

    if not text:
        return render_template('error.html', message="No text was provided. Please paste some text and try again."), 400

    try:
        result = ai_summary(text)
    except Exception as e:
        print(f"ERROR during summarization: {e}")
        return render_template('error.html', message="Summarization failed. Please try again with different text."), 500

    return render_template(
        'result.html',
        overall_summary=result["overall_summary"],
        paragraphs=result["paragraphs"],
        key_points=result["key_points"],
    )


@app.route('/upload', methods=['POST'])
def upload():
    """Handle PDF file upload, extract text, and return summarized result."""
    print("STEP 1: Upload route entered")

    # Validate file presence
    if 'pdf_file' not in request.files:
        return render_template('error.html', message="No file was included in the request."), 400

    file = request.files['pdf_file']

    if file.filename == '':
        return render_template('error.html', message="No file was selected. Please choose a PDF and try again."), 400

    if not file.filename.lower().endswith('.pdf'):
        return render_template('error.html', message="Only PDF files are supported. Please upload a valid PDF."), 400

    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(filepath)
        print(f"STEP 2: File saved → {filepath}")
    except Exception as e:
        print(f"ERROR saving file: {e}")
        return render_template('error.html', message="Failed to save the uploaded file. Please try again."), 500

    # Extract text from PDF
    try:
        text = extract_text(filepath)
        print(f"STEP 3: Text extracted — {len(text.split())} words")
    except Exception as e:
        print(f"ERROR extracting text: {e}")
        return render_template('error.html', message="Could not extract text from the PDF. The file may be scanned or corrupted."), 500

    if not text.strip():
        return render_template('error.html', message="No readable text found in the PDF. Please try a different file."), 400

    # Run summarization
    try:
        result = ai_summary(text)
        print("STEP 4: Summarization complete")
    except Exception as e:
        print(f"ERROR during summarization: {e}")
        return render_template('error.html', message="Summarization failed. Please try again with a different file."), 500

    return render_template(
        'result.html',
        overall_summary=result["overall_summary"],
        paragraphs=result["paragraphs"],
        key_points=result["key_points"],
    )


@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    """Generate and return a downloadable PDF from the summarized result."""
    overall_summary = request.form.get('overall_summary', '')
    paragraphs      = request.form.getlist('paragraphs')
    key_points      = request.form.getlist('key_points')

    if not overall_summary and not paragraphs and not key_points:
        return render_template('error.html', message="Nothing to download. Please generate a summary first."), 400

    try:
        return generate_pdf(overall_summary, paragraphs, key_points)
    except Exception as e:
        print(f"ERROR generating PDF: {e}")
        return render_template('error.html', message="PDF generation failed. Please try again."), 500


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)