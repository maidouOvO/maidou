from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import psycopg
import tempfile
import os
import sys

# Add parent directory to Python path to import PDFProcessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pdf_processor import PDFProcessor

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile,
    width: float = Form(...),
    height: float = Form(...)
):
    try:
        # Create temporary file for PDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            # Initialize PDFProcessor with the temporary file
            processor = PDFProcessor(temp_file.name, width, height)
            results_df = processor.process_pdf()

            # Create temporary directory for annotated PDF
            temp_dir = tempfile.mkdtemp()
            book_name = os.path.splitext(file.filename)[0]  # Extract name without extension
            annotated_pdf_path = os.path.join(temp_dir, f"{book_name}_annotated.pdf")
            processor.annotate_text_boxes(annotated_pdf_path)

            # Convert results to JSON-serializable format
            results = results_df.to_dict(orient='records')
            processor.close()
            os.unlink(temp_file.name)

            return {
                "success": True,
                "results": results,
                "annotated_pdf_path": annotated_pdf_path
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
