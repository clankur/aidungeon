# %%
import fitz

flags = fitz.TEXT_PRESERVE_LIGATURES


# %%
def read_pdf(path: str, page_range: tuple[int, int] = None) -> str:
    text = ""
    try:
        doc = fitz.open(path)
        # Define the flags for text extraction

        if page_range is None:
            for page in doc:
                # Use the flags in get_text()
                text += page.get_text(flags=flags)
        else:
            start_page = max(0, page_range[0] - 1)
            end_page = min(len(doc), page_range[1])  # Exclusive index for range/slice
            for i in range(start_page, end_page):
                page = doc[i]
                # Use the flags in get_text()
                text += page.get_text(flags=flags)
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading {path} with PyMuPDF: {e}")
        return None


# %%
# Example usage:
if __name__ == "__main__":
    pdf_file = "data/chap12.pdf"
    extracted_text = read_pdf(pdf_file)
    if extracted_text:
        print(extracted_text)
    # %%
    from storyteller import Extractor

    # %%
    extractor = Extractor(model_name="gemini-2.5-flash-preview-04-17")
    base_triples = extractor.extract(extracted_text)
    base_triples

    # %%
