import os
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def ingestion_func():
    print("INGESTING DOCUMENTS.....")
    
    dir_path = r"C:\Users\saksh\Documents\projects\Production RAG\dataset"
    dir_list = os.listdir(path=dir_path)
    file_itt = []

    for file in dir_list:
        file_path = os.path.join(dir_path, file)
        file_itt.append(file_path)

    loader = DoclingLoader(file_path=file_itt, converter=doc_converter, export_type=ExportType.MARKDOWN)
    document = loader.load()

    return document