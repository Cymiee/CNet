import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnet import MNISTData, matmul, relu, randn, cross_entropy_loss, save_weights, load_weights

LR      = 0.01
EPOCHS  = 3
H       = 128
IN_DIM  = 784
CLASSES = 10

def train(data, w1, w2):
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct    = 0

        for idx in range(data.count):
            img   = data.get_image_tensor(idx)   # (1, 784)
            label = data.get_label(idx)

            h      = relu(matmul(img, w1))        # (1, H)
            logits = matmul(h, w2)                # (1, CLASSES)

            total_loss += cross_entropy_loss(logits, label)
            correct    += int(logits.argmax() == label)

            logits.backward_grad_set()
            w1.sgd_step(LR)
            w2.sgd_step(LR)
            w1.zero_grad()
            w2.zero_grad()
            del logits, h, img  # free intermediate C tensors immediately

            if (idx + 1) % 5000 == 0:
                print(f"  epoch {epoch+1} [{idx+1:>5}/{data.count}]  "
                      f"loss={total_loss/(idx+1):.4f}  "
                      f"acc={correct/(idx+1):.3f}")

        print(f"Epoch {epoch+1}: loss={total_loss/data.count:.4f}  "
              f"acc={correct/data.count:.3f}")

def evaluate(data, w1, w2):
    correct = 0
    for idx in range(data.count):
        img   = data.get_image_tensor(idx)
        label = data.get_label(idx)
        h      = relu(matmul(img, w1))
        logits = matmul(h, w2)
        correct += int(logits.argmax() == label)
        del logits, h, img
    print(f"Accuracy: {correct}/{data.count} = {correct/data.count:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images')
    parser.add_argument('labels')
    parser.add_argument('--save', metavar='FILE', help='save weights after training')
    parser.add_argument('--load', metavar='FILE', help='load weights instead of training')
    parser.add_argument('--eval', action='store_true', help='evaluate instead of train (use with --load)')
    args = parser.parse_args()

    # He init: scale = sqrt(2 / fan_in)
    w1 = randn((IN_DIM, H),       requires_grad=True, scale=(2.0 / IN_DIM) ** 0.5)
    w2 = randn((H,      CLASSES), requires_grad=True, scale=(2.0 / H)      ** 0.5)

    if args.load:
        load_weights(args.load, [w1, w2])
        print(f"Loaded weights from {args.load}")

    data = MNISTData(args.images, args.labels)
    print(f"Loaded {data.count} samples ({data.rows}x{data.cols})")

    if args.eval or args.load:
        evaluate(data, w1, w2)
    else:
        train(data, w1, w2)
        if args.save:
            save_weights(args.save, [w1, w2])
            print(f"Saved weights to {args.save}")

if __name__ == "__main__":
    main()
