from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes





def preprocess_image(image):
    """Preprocess an image: resize and convert to grayscale."""
    fixed_width = 800
    height, width = image.shape[:2]
    scale = fixed_width / width
    resized_image =  cv2.resize(image, (fixed_width, int(height * scale)))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return resized_image, gray_image


def match_keypoints(img1, img2):
    """Find keypoints and matches between two images."""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches, keypoints1, keypoints2


def stitch_images_homography(img1, img2):
    """Stitch two images using homography."""
    good_matches, kp1, kp2 = match_keypoints(img1, img2)

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography matrix
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp the second image
        height, width = img1.shape[:2]
        stitched_img = cv2.warpPerspective(img2, H, (width * 2, height))
        stitched_img[0:height, 0:width] = img1

        return stitched_img
    else:
        raise ValueError("Not enough matches found for stitching!")


def crop_black_borders(image):
    """Crop black borders from the stitched image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return image[y:y+h, x:x+w]


@app.route('/stitch', methods=['POST'])
def stitch_images():
    """Endpoint to stitch images into a panorama."""
    data = request.json
    images_data = data.get('images', [])

    if len(images_data) < 2:
        return jsonify({"error": "At least two images are required for stitching"}), 400

    images = []
    for img_data in images_data:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img = Image.open(BytesIO(img_bytes))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed_img, _ = preprocess_image(img_cv)
        images.append(processed_img)

    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched = stitcher.stitch(images)

        if status != cv2.Stitcher_OK:
            # Attempt custom stitching with homography
            stitched = images[0]
            for i in range(1, len(images)):
                stitched = stitch_images_homography(stitched, images[i])

        stitched = crop_black_borders(stitched)

        # Encode the stitched image as JPEG
        _, buffer = cv2.imencode('.jpg', stitched)
        stitched_image_bytes = BytesIO(buffer.tobytes())

        return send_file(stitched_image_bytes, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": f"Image stitching failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)

