import os
import cv2
import uuid
import tempfile
import atexit
import streamlit as st
from AI_integrated_Model import SegmentationEngine, BrailleClassifier, BrailleImage
# 
# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Temporary directory for uploads and processed files
tempdir = tempfile.TemporaryDirectory()

# Clean up temp directory on exit
atexit.register(tempdir.cleanup)

# Utility: Check if uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def main():
    st.title("Optical Braille Recognition Demo")

    st.write("Upload an image file (png, jpg, jpeg) containing Braille text to extract the text.")

    uploaded_file = st.file_uploader("Choose an image file", type=list(ALLOWED_EXTENSIONS))

    if uploaded_file is not None:
        filename = ''.join(str(uuid.uuid4()).split('-'))
        raw_path = os.path.join(tempdir.name, filename)

        # Save uploaded file to temp directory
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process image
        classifier = BrailleClassifier()
        img = BrailleImage(raw_path)
        for letter in SegmentationEngine(image=img):
            letter.mark()
            classifier.push(letter)

        # Save processed image
        processed_path = os.path.join(tempdir.name, f"{filename}-proc.png")
        cv2.imwrite(processed_path, img.get_final_image())

        # Display original and processed images
        st.subheader("Original Image")
        st.image(uploaded_file, use_column_width=True)

        st.subheader("Processed Image")
        processed_image = cv2.imread(processed_path)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image, use_column_width=True)

        # Display extracted Braille text
        st.subheader("Extracted Braille Text")
        st.text_area("", classifier.digest(), height=150)

        # Clean up raw uploaded file
        os.unlink(raw_path)

    else:
        st.image("samples/cover.jpg", caption="Cover Image", use_column_width=True)

if __name__ == "__main__":
    main()
