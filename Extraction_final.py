import os
import requests
import pdfplumber
import json
import fitz  # PyMuPDF
import re
from PIL import Image
from io import BytesIO
import numpy as np

# Configurations
MIN_WIDTH = 50
MIN_HEIGHT = 50
MIN_FILE_SIZE = 500  # Minimum file size in bytes (e.g., 500 bytes)
RELEVANT_KEYWORDS = ["Figure", "Photo", "Graph", "Illustration", "Image"]

# Check if an image is blank or has minimal content
def is_blank_image(image_bytes):
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("L")  # Convert to grayscale
        img_array = np.array(img)
        pixel_variance = np.var(img_array)
        return pixel_variance < 10  # Adjust threshold as needed

# Step 1: Download PDFs and Extract Text
def download_and_extract_text_with_images(technologies, base_url, download_dir, output_json, image_folder):
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    data = []

    for tech, filename in technologies.items():
        pdf_url = base_url.format(filename)
        pdf_filename = os.path.join(download_dir, f"{tech}.pdf")
        tech_image_folder = os.path.join(image_folder, tech)
        os.makedirs(tech_image_folder, exist_ok=True)
        
        try:
            # Download the PDF
            print(f"Downloading {tech}...")
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()
            with open(pdf_filename, "wb") as file:
                file.write(response.content)

            # Extract text and images
            print(f"Processing {tech}...")
            text_content = ""
            pdf_document = fitz.open(pdf_filename)
            processed_images = set()  # Track processed xrefs
            images_for_json = []

            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                page_text = page.get_text()
                if page_text:  # Ensure the page has text
                    # Clean text: remove excessive whitespace, URLs, and unwanted newlines
                    page_text = re.sub(r"http\S+|www\S+", "", page_text)  # Remove URLs
                    page_text = " ".join(page_text.split())  # Remove excessive whitespace
                    text_content += page_text + "\n"

                # Extract images
                images = page.get_images(full=True)
                for img_index, img in enumerate(images, start=1):
                    xref = img[0]
                    if xref in processed_images:
                        continue  # Skip duplicate images
                    processed_images.add(xref)

                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width, height = base_image.get("width", 0), base_image.get("height", 0)

                    # Filter out small, blank, or irrelevant images
                    if width < MIN_WIDTH or height < MIN_HEIGHT or len(image_bytes) < MIN_FILE_SIZE:
                        continue
                    if is_blank_image(image_bytes):
                        continue

                    # Save the image
                    image_filename = f"page_{page_number + 1}_img_{img_index}.{image_ext}"
                    image_path = os.path.join(tech_image_folder, image_filename)
                    with open(image_path, "wb") as image_file:
                        image_file.write(image_bytes)
                    images_for_json.append(os.path.join(tech, image_filename))
                    # Add a reference to the content
                    text_content += f"\n[IMAGE: {image_filename}]\n"

            pdf_document.close()
            # Append to data
            data.append({
                "title": tech,
                "content": text_content.strip(),
                "images": images_for_json
            })
            print(f"Successfully processed {tech}.")
        
        except requests.exceptions.RequestException as req_err:
            print(f"Network error for {tech}: {req_err}")
        except Exception as e:
            print(f"Error processing {pdf_filename}: {e}")
            continue

    # Save data to JSON
    with open(output_json, "w") as outfile:
        json.dump(data, outfile, indent=4)
    print(f"Data extracted and saved to {output_json}.")

# Example Usage
if __name__ == "__main__":
    # PDF information
    technologies = {
    "2Dand3DImagingTechnologyIT": "2Dand3DImagingTechnologyIT",
    "CircularTextureMeterCTM": "CircularTextureMeterCTM",
    "ElectricalResistivityER": "ElectricalResistivityER",
    "FallingWeightDeflectometerFWD": "FallingWeightDeflectometerFWD",
    "GalvanostaticPulseMeasurementGPM": "GalvanostaticPulseMeasurementGPM",
    "GeorgiaFaultmeterGFM": "GeorgiaFaultmeterGFM",
    "GroundPenetratingRadarAsphaltCompactionAssessment": "GPRAsphaltCompactionAssessment",
    "GroundPenetratingRadarAsphaltDefects": "GPRAsphaltDefects",
    "GroundPenetratingRadarAsphaltThicknessAssessment": "GPRAsphaltThicknessAssessment",
    "GroundPenetratingRadarBaseSubbaseSubgradeDefects": "GPRBaseSubbaseSubgradeDefects",
    "GroundPenetratingRadarBaseSubbaseSubgradeThickness": "GPRBaseSubbaseSubgradeThickness",
    "GroundPenetratingRadarConcretePavementDefects": "GPRConcretePavementDefects",
    "GroundPenetratingRadarThicknessAndDowelLocation": "GPRThickness038DowelLocation",  # Adjusted based on provided URL
    "HalfCellPotentialHCP": "Half-CellPotentialHCP",
    "ImpactEchoIE": "ImpactEchoIE",
    "ImpulseResponseIR": "ImpulseResponseIR",
    "InertialProfilerPavementIP": "InertialProfiler8211PavementIP",
    "InfraredThermographyIT": "InfraredThermographyIT",
    "IowaVacuumJointSealTesterIAVAC": "IowaVacuumJointSealTesterIA-VAC",  # Adjusted based on provided URL
    "LockedWheelTesterLWT": "Locked-WheelTesterLWT",
    "MagneticPulseInductionMPI": "MagneticPulseInductionMPI",
    "MagnetometerMM": "MagnetometerMM",
    "MicrophonesMP": "MicrophonesMP",
    "NuclearDensityGaugeNDG": "NuclearDensityGaugeNDG",
    "PhotometricStereoImagingPSI": "PhotometricStereoImagingPSI",
    "SandPatchTestSPT": "SandPatchTestSPT",
    "Sounding": "Sounding",
    "StraightedgePavementApplicationSEP": "Straightedge8211PavementApplicationSE-P",  # Adjusted based on provided URL
    "StraightedgeSE": "StraightedgeSE",
    "TrafficSpeedDeflectionDevicesTSDDs": "TrafficSpeedDeflectionDevicesTSDDs",
    "UltrasonicSurfaceWaveUSW": "UltrasonicSurfaceWaveUSW",
    "UltrasonicTomographyUST": "UltrasonicTomographyUST"
    }
    base_url = "https://infotechnology.fhwa.dot.gov/wp-content/themes/nde/inc/mpdf-development/Generatedpdfs/{}.pdf"
    download_dir = "pdf_downloads_final"
    output_json = "pavement_data.json"
    image_folder = "images"

    # Step 1: Download, Extract Text, and Filter Images
    download_and_extract_text_with_images(technologies, base_url, download_dir, output_json, image_folder)
