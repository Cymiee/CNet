import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnet import MNISTData, matmul, relu, randn, cross_entropy_loss

LR      = 0.01
EPOCHS  = 3
H       = 128
IN_DIM  = 784
CLASSES = 10

def main(img_path, lbl_path):
    data = MNISTData(img_path, lbl_path)
    print(f"Loaded {data.count} samples ({data.rows}x{data.cols})")

    # He init: scale = sqrt(2 / fan_in)
    w1 = randn((IN_DIM, H),      requires_grad=True, scale=(2.0 / IN_DIM) ** 0.5)
    w2 = randn((H,      CLASSES), requires_grad=True, scale=(2.0 / H)      ** 0.5)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct    = 0

        for idx in range(data.count):
            img   = data.get_image_tensor(idx)   # (1, 784)
            label = data.get_label(idx)

            h      = relu(matmul(img, w1))        # (1, H)
            logits = matmul(h, w2)                # (1, CLASSES)

            # Sets logits.grad to softmax gradient, returns scalar loss
            loss = cross_entropy_loss(logits, label)
            total_loss += loss
            correct    += int(logits.argmax() == label)

            # Propagate from logits (grad already set by cross_entropy_loss)
            logits.backward_grad_set()

            w1.sgd_step(LR)
            w2.sgd_step(LR)
            w1.zero_grad()
            w2.zero_grad()

            if (idx + 1) % 5000 == 0:
                print(f"  epoch {epoch+1} [{idx+1:>5}/{data.count}]  "
                      f"loss={total_loss/(idx+1):.4f}  "
                      f"acc={correct/(idx+1):.3f}")

        print(f"Epoch {epoch+1}: loss={total_loss/data.count:.4f}  "
              f"acc={correct/data.count:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python demo.py <images-file> <labels-file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
