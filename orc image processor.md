# ImagePreprocessor Documentation - Simple Guide

## What This Class Does
The `ImagePreprocessor` class helps prepare images for better text recognition (OCR). It applies various filters and adjustments to make text clearer and easier to read by computer programs.

---

## Core Functions

### **1. `invert_image()`** - Flip Colors
- **What it does:** Makes black text white, and white background black
- **When to use:** Sometimes inverted images work better for text recognition
- **Variables:** None to change
- **Effect:** Complete color reversal

### **2. `rescale_image(scale_factor)`** - Resize Image
- **What it does:** Makes image bigger or smaller
- **Variables:** 
  - `scale_factor = 1.0` → Same size
  - `scale_factor = 2.0` → Double size
  - `scale_factor = 0.5` → Half size
- **Effect on text:** Bigger images often give better OCR results

### **3. `grayscale()`** - Remove Colors
- **What it does:** Converts colorful image to black, white, and gray only
- **Variables:** None to change
- **Effect:** Often improves text recognition, removes color distractions

### **4. `binarize_image(threshold)`** - Make Pure Black & White
- **What it does:** Converts gray pixels to either pure black or pure white
- **Variables:** 
  - `threshold = 127` → Medium sensitivity (recommended)
  - `threshold = 105` → More pixels become black (darker result)
  - `threshold = 150` → More pixels become white (lighter result)
- **Effect on text:** Can make text sharper but may lose thin letters if threshold is wrong

### **5. `noise_removal()`** - Clean Up Spots
- **What it does:** Removes small dots, specks, and unwanted marks
- **Variables:** Kernel sizes are hardcoded (1x1 pixels)
- **Effect:** Cleaner image but may remove small text details

### **6. `thin_font()` / `thick_font()`** - Adjust Text Thickness
- **What they do:** Make letters thinner or thicker
- **Variables:** Kernel size (2x2 pixels) controls how much change
- **Effect:** 
  - `thin_font()` → Better for thick, bold text
  - `thick_font()` → Better for very thin, light text

### **7. `deskew()`** - Straighten Tilted Text
- **What it does:** Rotates image to make text lines horizontal
- **Variables:** Automatically detects angle
- **Effect:** Improves OCR accuracy for crooked scanned documents

### **8. `remove_borders()`** - Crop to Content
- **What it does:** Cuts away empty white space around the text
- **Variables:** None to change
- **Effect:** Focuses OCR on actual content, may improve accuracy

### **9. `add_borders(border_size, color)`** - Add White Space
- **What it does:** Adds padding around the image
- **Variables:**
  - `border_size = 150` → 150 pixels of padding (default)
  - `color = [255,255,255]` → White padding (default)
- **Effect:** Sometimes OCR works better with white space around text

---

## Helper Functions

### **`display(image_path)`** - Show Image
- Shows the image on screen in actual size

### **`extract_fraktur_text(image_path)`** - Read German Text
- Uses special German Gothic font recognition
- Returns the text it found in the image

---

## Complete Processing Pipeline

### **`preprocess_pipeline()`** - Do Everything
Runs these steps in order:
1. Invert colors
2. Convert to grayscale  
3. Make black & white (binarize)
4. Remove noise
5. Remove borders
6. Add new borders

**Files saved:** Each step saves an image file so you can see the changes

---

## Key Tips for Better Results

**Threshold Value (most important):**
- Start with 127
- If text looks too thin → use lower number (like 105)
- If text looks too thick → use higher number (like 150)

**Processing Order Matters:**
- Each step can make the next step work better or worse
- Grayscale usually gives best OCR results
- Binarization can help but needs right threshold

**When to Stop:**
- Test OCR after each step
- Stop when you get good text recognition
- More processing isn't always better