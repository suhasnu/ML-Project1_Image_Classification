import os
import sys
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

def clean_image(image):
    cleaned = image.copy()
    h, w = cleaned.shape
    
    # For Horizontal Row
    for y in range(h):
        if np.mean(cleaned[y, :]) > 240 and np.std(cleaned[y, :]) < 10:
            if y > 0:
                cleaned[y, :] = cleaned[y-1, :]
            else:
                cleaned[y, :] = 0
                          
    #For Vertical Columns
    for x in range(w):
        if np.mean(cleaned[:, x]) > 240 and np.std(cleaned[:, x]) < 10:
            if x > 0:
                cleaned[:, x] = cleaned[:, x-1]
            else:
                cleaned[:, x] = 0

    return cleaned


def show_comparison(original, cleaned, label):
    plt.figure(figsize=(8,4))
    
    plt.subplot(1,2,1)
    plt.imshow(original, cmap='gray')
    plt.title(f"Original: {label}")
    plt.subplot(1,2,2)
    plt.imshow(cleaned, cmap='gray')
    plt.title("Fixed")
    plt.show()

def main():
    if len(sys.argv) < 7:
        print("Enter enough arguments!")
        sys.exit(1)
        
    image_directory = sys.argv[1]
    h = int(sys.argv[2])
    w = int(sys.argv[3])
    c = int(sys.argv[4])
    output_file = sys.argv[5]
    correct_problems = int(sys.argv[6])
    
    #Create two empty lists: one to hold the pixel data (images) and one for the class numbers (labels)  
    #Get all files and sort them alphabatically!!
    all_images = sorted(os.listdir(image_directory))  
    images = []
    labels = []
    shown_count = 0
    
    #Then we need to go through every files one by one.
    for fname in all_images: 
        if fname.endswith(".png"):
 
            fpath = os.path.join(image_directory, fname)
            label = int(fname.split("-")[0])
            
            # Open, Grayscale, and Resize to ensure consistency
            img_obj = Image.open(fpath).convert("L")
            img_obj = img_obj.resize((w, h))
            image = np.asarray(img_obj)
            
        
            if image.shape != (h, w): 
                continue            
            
            if correct_problems == 1:
                original = image.copy()
                image = clean_image(image)
                
                if not np.array_equal(original, image):
                    if shown_count < 10:
                        print(f"Visualizing {fname}")
                        show_comparison(original, image, label)
                        shown_count += 1
                        
                    # To show the demo!
                    if shown_count >= 10 and "junk" in output_file:
                        print("Demo Complete")
                        sys.exit(0)
                        
            images.append(image)
            labels.append(label)
            
    np.savez(output_file, images=np.array(images), labels=np.array(labels))
    print("Data.npz file saved successfully!")


if __name__ == "__main__":
    main()