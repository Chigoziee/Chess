from flask import Flask, request, jsonify
import io, os
import torch
from PIL import Image
from App import load_checkpoint, predict

app = Flask(__name__)

piece_label = {0: "Bishop", 1: "King", 2: "knight", 3: "pawn", 4: "Queen", 5: "Rook"}

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'
model = load_checkpoint('checkpoint.pth', map_location)


@app.route('/Classify', methods=['POST'])
def classify():
    try:
        # This ensures a file is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        # Get image file from request
        image_ = request.files['image']
        # Get image and its extension
        image_name, image_extension = os.path.splitext(image_.filename)
        # Ensure only image files are uploaded
        if image_extension not in ['.jpeg', '.jpg', '.png', '.webp', '.hdr', '.tiff', '.tif']:
            return jsonify({'error': 'Upload only image files'}), 400
        else:
            # Read image
            img = Image.open(io.BytesIO(image_.read()))

            # Get predictions
            topk_prob, topk_idx = predict(img, model, topk=1)

            piece = piece_label[topk_idx[0]].upper()
            if topk_prob[0] > 0.5:
                return jsonify({"Result": f'THIS IS A {piece}'})
            else:
                return jsonify({"error": "Please upload a clearer image!"})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
