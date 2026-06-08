import sys
import os
import argparse
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnet import (MNISTData, matmul, relu, randn, zeros,
                  cross_entropy_loss_batch, argmax_rows,
                  save_weights, load_weights, Adam)

LR         = 0.001
EPOCHS     = 3
H          = 512
IN_DIM     = 784
CLASSES    = 10
BATCH_SIZE = 64

def train(data, w1, b1, w2, b2):
    opt     = Adam([w1, b1, w2, b2], lr=LR)
    indices = list(range(data.count))

    for epoch in range(EPOCHS):
        random.shuffle(indices)
        total_loss = 0.0
        correct    = 0
        seen       = 0

        for start in range(0, data.count, BATCH_SIZE):
            batch_idx      = indices[start:start + BATCH_SIZE]
            imgs, labels   = data.get_batch_tensor(batch_idx)   # (B, 784)

            h      = relu(matmul(imgs, w1) + b1)                # (B, H)
            logits = matmul(h, w2) + b2                         # (B, CLASSES)

            batch_loss  = cross_entropy_loss_batch(logits, labels)
            preds       = argmax_rows(logits)
            B           = len(batch_idx)
            total_loss += batch_loss * B
            correct    += sum(p == l for p, l in zip(preds, labels))
            seen       += B

            logits.backward_grad_set()
            opt.step()
            opt.zero_grad()
            del logits, h, imgs

            if seen % 5000 < BATCH_SIZE:
                print(f"  epoch {epoch+1} [{seen:>5}/{data.count}]  "
                      f"loss={total_loss/seen:.4f}  "
                      f"acc={correct/seen:.3f}")

        print(f"Epoch {epoch+1}: loss={total_loss/data.count:.4f}  "
              f"acc={correct/data.count:.3f}")

def evaluate(data, w1, b1, w2, b2):
    correct = 0
    for start in range(0, data.count, BATCH_SIZE):
        batch_idx    = list(range(start, min(start + BATCH_SIZE, data.count)))
        imgs, labels = data.get_batch_tensor(batch_idx)
        h      = relu(matmul(imgs, w1) + b1)
        logits = matmul(h, w2) + b2
        preds  = argmax_rows(logits)
        correct += sum(p == l for p, l in zip(preds, labels))
        del logits, h, imgs
    print(f"Accuracy: {correct}/{data.count} = {correct/data.count:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images')
    parser.add_argument('labels')
    parser.add_argument('--save', metavar='FILE', help='save weights after training')
    parser.add_argument('--load', metavar='FILE', help='load weights instead of training')
    parser.add_argument('--eval', action='store_true', help='evaluate instead of train (use with --load)')
    args = parser.parse_args()

    # He init for weights; biases start at zero
    w1 = randn((IN_DIM, H),       requires_grad=True, scale=(2.0 / IN_DIM) ** 0.5)
    b1 = zeros((1, H),            requires_grad=True)
    w2 = randn((H,      CLASSES), requires_grad=True, scale=(2.0 / H)      ** 0.5)
    b2 = zeros((1,      CLASSES), requires_grad=True)

    if args.load:
        load_weights(args.load, [w1, b1, w2, b2])
        print(f"Loaded weights from {args.load}")

    data = MNISTData(args.images, args.labels)
    print(f"Loaded {data.count} samples ({data.rows}x{data.cols})")

    if args.eval or args.load:
        evaluate(data, w1, b1, w2, b2)
    else:
        train(data, w1, b1, w2, b2)
        if args.save:
            save_weights(args.save, [w1, b1, w2, b2])
            print(f"Saved weights to {args.save}")

if __name__ == "__main__":
    main()
