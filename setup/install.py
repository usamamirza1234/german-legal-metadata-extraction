import os
import urllib.request
import shutil


def install_german_fraktur():
    """Download and install German Fraktur language data for Tesseract"""

    # Your tessdata directory
    tessdata_dir = r"C:\Users\usama\AppData\Local\Programs\Tesseract-OCR\tessdata"

    print(f"Installing to: {tessdata_dir}")

    # URLs for German Fraktur language files
    files_to_download = {
        'deu_frak.traineddata': 'https://github.com/tesseract-ocr/tessdata/raw/main/deu_frak.traineddata',
        'deu_latf.traineddata': 'https://github.com/tesseract-ocr/tessdata/raw/main/deu_latf.traineddata'
    }

    for filename, url in files_to_download.items():
        file_path = os.path.join(tessdata_dir, filename)

        if os.path.exists(file_path):
            print(f"✓ {filename} already exists")
            continue

        try:
            print(f"📥 Downloading {filename}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"✓ {filename} installed successfully")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

    print("\n🔍 Checking installation:")

    # Verify files
    for filename in files_to_download.keys():
        file_path = os.path.join(tessdata_dir, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ {filename}: {size:.1f} MB")
        else:
            print(f"❌ {filename}: Not found")


def check_tessdata_directory():
    """Check what's in your tessdata directory"""
    tessdata_dir = r"C:\Users\usama\AppData\Local\Programs\Tesseract-OCR\tessdata"

    print(f"📁 Tessdata directory: {tessdata_dir}")

    if not os.path.exists(tessdata_dir):
        print("❌ Tessdata directory not found!")
        return

    files = [f for f in os.listdir(tessdata_dir) if f.endswith('.traineddata')]
    print(f"\n📋 Available language files ({len(files)}):")

    for file in sorted(files):
        size = os.path.getsize(os.path.join(tessdata_dir, file)) / (1024 * 1024)
        print(f"  {file:<20} ({size:.1f} MB)")

    # Check for German files specifically
    german_files = [f for f in files if 'deu' in f]
    print(f"\n🇩🇪 German language files: {len(german_files)}")
    for file in german_files:
        print(f"  ✓ {file}")


if __name__ == "__main__":
    print("=== GERMAN FRAKTUR INSTALLATION ===")

    # First check what you have
    check_tessdata_directory()

    print("\n" + "=" * 40)

    # Install missing files
    install_german_fraktur()

    print("\n" + "=" * 40)

    # Check again
    check_tessdata_directory()