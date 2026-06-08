import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PIL import Image
except ImportError:
    print("Pillow is required:  pip install Pillow")
    sys.exit(1)

from cnet import matmul, relu, zeros, load_weights, softmax, Tensor

H       = 512
IN_DIM  = 784
CLASSES = 10

def preprocess(image_path):
    img = Image.open(image_path).convert('L')       # grayscale

    # Auto-invert if background is light (drawing apps give black on white)
    pixels = list(img.getdata())
    if sum(pixels) / len(pixels) > 127:
        img = img.point(lambda p: 255 - p)

    # Crop to bounding box and re-center with padding — matches MNIST preprocessing
    bbox = img.getbbox()
    if bbox:
        digit  = img.crop(bbox)
        pad    = max(digit.size) // 5
        padded = Image.new('L', (digit.width + pad*2, digit.height + pad*2), 0)
        padded.paste(digit, (pad, pad))
        img    = padded

    img = img.resize((28, 28), Image.LANCZOS)
    return [p / 255.0 for p in img.getdata()]

def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit")
    parser.add_argument('weights',    help='weights file from demo.py --save')
    parser.add_argument('image',      help='image of a handwritten digit (PNG/JPG)')
    args = parser.parse_args()

    w1 = zeros((IN_DIM, H))
    b1 = zeros((1, H))
    w2 = zeros((H, CLASSES))
    b2 = zeros((1, CLASSES))
    load_weights(args.weights, [w1, b1, w2, b2])

    flat       = preprocess(args.image)
    img_tensor = Tensor([flat])                     # (1, 784)

    h      = relu(matmul(img_tensor, w1) + b1)      # (1, H)
    logits = matmul(h, w2) + b2                     # (1, CLASSES)

    probs  = softmax(logits)
    top3   = sorted(range(CLASSES), key=lambda i: probs[i], reverse=True)[:3]

    print(f"Prediction: {top3[0]}")
    print("Top 3:")
    for rank, digit in enumerate(top3, 1):
        print(f"  {rank}. digit {digit}  ({probs[digit]*100:.1f}%)")

if __name__ == "__main__":
    main()
