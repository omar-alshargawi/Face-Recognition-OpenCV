# === Webcam Capture Function ===
from IPython.display import display, Javascript
from google.colab.output import eval_js
import base64, os, shutil, cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
      async function takePhoto(quality) {
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = '📸 Take Photo';
        div.appendChild(capture);

        const video = document.createElement('video');
        video.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();

        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
        await new Promise((resolve) => capture.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getVideoTracks()[0].stop();
        div.remove();

        const dataUrl = canvas.toDataURL('image/jpeg', quality);
        return dataUrl;
      }
      takePhoto(%s);
    ''' % quality)

    display(js)
    data = eval_js("takePhoto({quality: %s})" % quality)
    binary = base64.b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# === Step 2: Prepare Dataset ===
dataset_dir = "face_dataset"
if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
os.makedirs(dataset_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
user_id = 1

print("📸 Capture 5 face images for training:")
for i in range(10):
    filename = take_photo(f"{dataset_dir}/User.{user_id}.{i+1}.jpg")
    print(f"Saved: {filename}")

# === Step 3: Train the LBPH Recognizer ===
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        pil_img = Image.open(image_path).convert('L')
        img_np = np.array(pil_img, 'uint8')
        face_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_np)
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(face_id)
    return face_samples, ids

faces, ids = get_images_and_labels(dataset_dir)
recognizer.train(faces, np.array(ids))
print("✅ Training complete")

# === Step 4: Test Recognition ===
test_img_path = take_photo('test.jpg')
print("📷 Testing on:", test_img_path)

img = cv2.imread(test_img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)

for (x, y, w, h) in faces:
    roi = gray[y:y+h, x:x+w]
    predicted_id, confidence = recognizer.predict(roi)
    label = f"User {predicted_id} ({round(100 - confidence, 1)}% confidence)"
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Face Recognition Result")
plt.show()
