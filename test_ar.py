import os
import pdfplumber

docs_dir = 'member1-yield-model/data/raw/source_docs'
pdfs_to_test = [
    'Final_Annual_Report_2021_22_14_12_2022_pdf414.pdf',
    'Annual_Report_2020_2021_pdf1655.pdf'
]

for filename in pdfs_to_test:
    filepath = os.path.join(docs_dir, filename)
    if not os.path.exists(filepath): continue
    print(f"\n{'='*50}\nTesting PDF: {filename}\n{'='*50}")
    
    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total Pages: {total_pages}")
            
            readable_pages = []
            target_pages = []
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    readable_pages.append(i + 1)
                    lower_text = text.lower()
                    if 'production' in lower_text and 'area' in lower_text and ('yield' in lower_text or 'hectare' in lower_text):
                        if 'assam' in lower_text or 'darjeeling' in lower_text:
                            target_pages.append(i + 1)
                        
            print(f"Status: MACHINE-READABLE text found on {len(readable_pages)}/{total_pages} pages.")
            if target_pages:
                print(f"Keywords (Area/Production/Yield + Regions) found on pages: {target_pages[:5]}...")
                first_match = target_pages[0] - 1
                
                print(f"\n--- Extracting Table from Page {first_match + 1} ---")
                tables = pdf.pages[first_match].extract_tables()
                if tables:
                    for idx, table in enumerate(tables):
                        print(f"Table {idx+1} (Rows: {len(table)}):")
                        for row in table[:10]:
                            print([str(cell)[:20].replace('\n', ' ') if cell else '' for cell in row])
                else:
                    print("No structured tables extracted.")
            else:
                print("Target keywords NOT found in text.")
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")
