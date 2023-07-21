from img2table.document import Image
from img2table.ocr import TesseractOCR
import pandas as pd

def extract_table_as_json(image_path):
    # Instantiation of the image
    img = Image(src = image_path)
    print(type(img))
    ocr = TesseractOCR(lang="eng")
    # We can also create an excel file with the tables
    img.to_xlsx('./tables.xlsx', ocr=ocr)
    df = pd.read_excel('./tables.xlsx')
    # json_data = df.to_json(orient='records', indent=4)
    return df