import cv2
import numpy as np
import os

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("🎨 Choose an option:")
print("1 - Capture photo from camera")
print("2 - Upload an image file")
choice = input("Enter 1 or 2: ")

captured = False

if choice == '1':
    # --- Webcam Capture Mode ---
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            padding_h = int(h * 1.0)
            padding_w = int(w * 0.3)
            x1 = max(x - padding_w, 0)
            y1 = max(y - int(h * 0.3), 0)
            x2 = min(x + w + padding_w, frame.shape[1])
            y2 = min(y + h + padding_h, frame.shape[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Press 'c' to Confirm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                cropped = frame[y1:y2, x1:x2]
                cv2.imwrite("confirmed_face.jpg", cropped)
                captured = True
                break

        cv2.imshow("Face + Neck Detection", frame)

        if captured or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif choice == '2':
    # --- Upload Mode ---
    path = input("📂 Enter image file path: ").strip('"')
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            cv2.imwrite("confirmed_face.jpg", img)
            captured = True
        else:
            print("❌ Couldn't read image. Try again.")
    else:
        print("❌ File not found. Please check the path.")

else:
    print("❌ Invalid choice.")

# --- Step 2: Realistic sketch generation ---
if captured:
    img = cv2.imread("confirmed_face.jpg")
    resized_img = cv2.resize(img, (400, 700))

    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray

    blur = cv2.GaussianBlur(inv, (701, 701), 0)
    raw_sketch = cv2.divide(gray, 255 - blur, scale=256.0)

    alpha = 1.6
    beta = -30
    sketch = cv2.convertScaleAbs(raw_sketch, alpha=alpha, beta=beta)

    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    edges = 255 - edges
    pencil_sketch = cv2.min(sketch, edges)

    sketch_normalized = pencil_sketch / 255.0
    sketch_color = cv2.merge([sketch_normalized] * 3)

    def apply_color_tint(color_bgr):
        color_array = np.array(color_bgr) / 255.0
        tinted = sketch_color * color_array
        return (tinted * 255).astype(np.uint8)

    current_sketch = (sketch_color * 255).astype(np.uint8)

    print("\n🎨 Press keys to switch color:")
    print("   'r' - Red sketch")
    print("   'g' - Green sketch")
    print("   'b' - Blue sketch")
    print("   'o' - Original grayscale sketch")
    print("   's' - Save sketch")
    print("   'q' - Quit\n")

    while True:
        cv2.imshow("Realistic Pencil Sketch", current_sketch)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("final_sketch.jpg", current_sketch)
            print("✅ Saved as 'final_sketch.jpg'")
        elif key == ord('r'):
            current_sketch = apply_color_tint((0, 0, 255))
            print("🖼️ Red sketch applied")
        elif key == ord('g'):
            current_sketch = apply_color_tint((0, 255, 0))
            print("🖼️ Green sketch applied")
        elif key == ord('b'):
            current_sketch = apply_color_tint((255, 0, 0))
            print("🖼️ Blue sketch applied")
        elif key == ord('o'):
            current_sketch = (sketch_color * 255).astype(np.uint8)
            print("🖼️ Original grayscale sketch")

    cv2.destroyAllWindows()
else:
    print("❌ No image was processed.") 