from langchain_community.document_loaders.base import BaseLoader
import xml.etree.ElementTree as ET
from langchain_core.documents import Document


class HDevLoader(BaseLoader):

  def __init__(self, file_path: str):
    self.file_path = file_path

  def lazy_load(self):
      try:
          # Explicitly read with UTF-8 and ignore weird characters to prevent encoding crashes
          with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
              xml_content = f.read()

          tree = ET.fromstring(xml_content)
          code_lines = [
              elem.text.strip()
              for elem in tree.iter()
              if elem.tag.endswith("l") and elem.text
          ]
          clean_code = "\n".join(code_lines)

          if clean_code:
              yield Document(
                  page_content=clean_code,
                  metadata={"source": self.file_path, "doc_type": "code"},
              )
      except Exception as e:
          print(f"Error parsing {self.file_path}: {e}")