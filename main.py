import os
from Colour_detection.detect_colours import detect_colors

def run_color_detection():
    """
    Runs the color detection module with the specified image.
    """
    try:
        # Define paths
        image_path = "Colour_detection/image.jpg"  # Update to your image path
        output_dir = "color_detection_output"  # Output directory

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found. Please upload 'image.jpg' to Colour_detection folder.")
            return

        # Run color detection
        detect_colors(image_path, output_dir)
        print("Color detection completed successfully.")

    except Exception as e:
        print(f"Error in run_color_detection: {str(e)}")

def main():
    """
    Main function to orchestrate the project workflow.
    """
    print("Starting Traffic Accident Prediction Project - Level 2 Enhancement")
    print("Running Color Detection Module...")
    
    # Run color detection
    run_color_detection()
    
    # Add more modules or functionalities here (e.g., accident prediction)
    # Example: import train_model and call it if needed
    print("Project execution completed.")

if __name__ == "__main__":
    main()
