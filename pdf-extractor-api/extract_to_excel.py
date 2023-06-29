import io
import os
import pandas as pd
import shutil
import tempfile
import uuid
import warnings
warnings.filterwarnings("ignore")
# from openpyxl import load_workbook


from flask import Flask, request, send_file
from flask_cors import CORS
from img2table.document import PDF
from img2table.ocr import TesseractOCR

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

tesseract_ocr = TesseractOCR(n_threads=1, lang="eng")

@app.route("/")
def index():
    return "PDF to Excel Conversion"

@app.route("/convert", methods=["POST"])
def convert_to_excel():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save the uploaded PDF file
    uid = uuid.uuid4().hex
    print(file.filename)
    file_name_str  = file.filename.split(".pdf")[0]
    upload_path = f"{app.config['UPLOAD_FOLDER']}/{file_name_str}.pdf"
    file.save(upload_path)

    # Perform PDF to Excel conversion
    pdf = PDF(src=upload_path)
    extracted_tables = pdf.extract_tables(
        ocr=tesseract_ocr,
        implicit_rows=False,
        borderless_tables=False,
        min_confidence=50
    )

    # Create a temporary directory to store Excel file
    with tempfile.TemporaryDirectory() as temp_dir:
        excel_path = f"{temp_dir}/output.xlsx"
        print(excel_path)
        # Convert tables to Excel
        excel_writer = pd.ExcelWriter(excel_path)
        for page, tables in extracted_tables.items():
            merged_data = pd.DataFrame()
            for idx, table in enumerate(tables):
                table_title = table.title
                table_content = table.df.to_csv(index=False)
                table_content_df = pd.read_csv(io.StringIO(table_content))
                merged_data = merged_data.append(
                    {"Page": page + 1, "Title": f"page_{page}_table{idx}"}, ignore_index=True
                )
                merged_data = merged_data.append(table_content_df, ignore_index=True)
                merged_data = merged_data.append(pd.Series(), ignore_index=True)
                if table_title is not None:
                    merged_data = merged_data.append(
                        {"Page": page + 1, "Title": f"page_{page}_paragraph{idx}", "Content": table_title},
                        ignore_index=True,
                    )
                    merged_data = merged_data.append(pd.Series(), ignore_index=True)
            merged_data.to_excel(excel_writer, sheet_name=f"Page {page + 1}", index=False)

        excel_writer.save()

        # Generate a downloadable file
        filename = f"{file_name_str}.xlsx"
        shutil.move(excel_path, f"{app.config['UPLOAD_FOLDER']}/{filename}")

        # return send_file(excel_writer,as_attachment= True)
        return send_file(
    f"{app.config['UPLOAD_FOLDER']}/{filename}",
    as_attachment=True,
    download_name="output.xlsx",
    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
