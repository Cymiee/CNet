import sys
import os
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnet import matmul, relu, zeros, load_weights, softmax, Tensor

CANVAS_SIZE = 280   # 10x MNIST (28x28)
BRUSH       = 18    # scales to ~1.8px at 28x28 — thick enough to recognise
H           = 128
IN_DIM      = 784
CLASSES     = 10
DEBOUNCE_MS = 40    # re-run inference at most every 40ms while drawing


class DrawApp:
    def __init__(self, root, w1, b1, w2, b2):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        self._after_id = None
        self._last_xy  = None

        root.title("CNet — Draw a Digit")
        root.resizable(False, False)
        root.configure(bg='#1e1e1e')

        # ── Drawing canvas ───────────────────────────────────────────────
        canvas_frame = tk.Frame(root, bg='#1e1e1e')
        canvas_frame.grid(row=0, column=0, padx=16, pady=16)

        tk.Label(canvas_frame, text="Draw here", bg='#1e1e1e', fg='#888',
                 font=('Helvetica', 11)).pack(anchor='w', pady=(0, 4))

        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_SIZE,
                                height=CANVAS_SIZE, bg='white',
                                cursor='crosshair', highlightthickness=0)
        self.canvas.pack()

        tk.Button(canvas_frame, text='Clear', command=self._clear,
                  bg='#333', fg='white', activebackground='#555',
                  activeforeground='white', relief='flat',
                  font=('Helvetica', 12), padx=16, pady=6).pack(pady=(10, 0))

        # PIL image — drawn in parallel, used for inference
        self._reset_pil()

        # ── Prediction panel ─────────────────────────────────────────────
        pred_frame = tk.Frame(root, bg='#1e1e1e')
        pred_frame.grid(row=0, column=1, padx=(0, 16), pady=16, sticky='n')

        tk.Label(pred_frame, text="Prediction", bg='#1e1e1e', fg='#888',
                 font=('Helvetica', 11)).pack(anchor='w')

        self.pred_label = tk.Label(pred_frame, text='?', bg='#1e1e1e',
                                   fg='#4fc3f7', font=('Helvetica', 80, 'bold'))
        self.pred_label.pack()

        tk.Label(pred_frame, text="Probabilities", bg='#1e1e1e', fg='#888',
                 font=('Helvetica', 11)).pack(anchor='w', pady=(8, 4))

        self._bars   = []
        self._pct_lbl = []
        for d in range(CLASSES):
            row = tk.Frame(pred_frame, bg='#1e1e1e')
            row.pack(fill='x', pady=1)
            tk.Label(row, text=str(d), width=2, bg='#1e1e1e', fg='white',
                     font=('Helvetica', 11, 'bold')).pack(side='left')
            bar = ttk.Progressbar(row, orient='horizontal', length=190,
                                  maximum=100, mode='determinate')
            bar.pack(side='left', padx=(4, 4))
            lbl = tk.Label(row, text='  0%', width=5, bg='#1e1e1e', fg='#aaa',
                           font=('Helvetica', 10), anchor='w')
            lbl.pack(side='left')
            self._bars.append(bar)
            self._pct_lbl.append(lbl)

        # ── Events ───────────────────────────────────────────────────────
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

    # ── Drawing ──────────────────────────────────────────────────────────

    def _on_drag(self, event):
        x, y = event.x, event.y
        r = BRUSH // 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill='black', outline='black')
        self._pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

        if self._last_xy:
            lx, ly = self._last_xy
            self.canvas.create_line(lx, ly, x, y, width=BRUSH,
                                    fill='black', capstyle=tk.ROUND)
            self._pil_draw.line([lx, ly, x, y], fill=0, width=BRUSH)

        self._last_xy = (x, y)
        self._schedule_inference()

    def _on_release(self, event):
        self._last_xy = None
        self._run_inference()

    # ── Inference ────────────────────────────────────────────────────────

    def _schedule_inference(self):
        if self._after_id:
            self.canvas.after_cancel(self._after_id)
        self._after_id = self.canvas.after(DEBOUNCE_MS, self._run_inference)

    def _run_inference(self):
        self._after_id = None
        small  = self._pil_img.resize((28, 28), Image.LANCZOS)
        pixels = list(small.getdata())           # 0 = black, 255 = white
        flat   = [(255 - p) / 255.0 for p in pixels]  # invert → MNIST convention

        if max(flat) < 0.05:                     # blank canvas
            self._set_blank()
            return

        img_t  = Tensor([flat])
        h      = relu(matmul(img_t, self.w1) + self.b1)
        logits = matmul(h, self.w2) + self.b2
        probs  = softmax(logits)

        pred   = max(range(CLASSES), key=lambda i: probs[i])
        self.pred_label.config(text=str(pred))

        for d in range(CLASSES):
            pct = probs[d] * 100
            self._bars[d]['value'] = pct
            self._pct_lbl[d].config(
                text=f'{pct:4.1f}%',
                fg='#4fc3f7' if d == pred else '#aaa'
            )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _reset_pil(self):
        self._pil_img  = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
        self._pil_draw = ImageDraw.Draw(self._pil_img)

    def _clear(self):
        self.canvas.delete('all')
        self._reset_pil()
        self._set_blank()

    def _set_blank(self):
        self.pred_label.config(text='?')
        for d in range(CLASSES):
            self._bars[d]['value'] = 0
            self._pct_lbl[d].config(text='  0%', fg='#aaa')


def main():
    parser = argparse.ArgumentParser(
        description='Draw a digit and see real-time CNet predictions')
    parser.add_argument('weights', help='weights file produced by demo.py --save')
    args = parser.parse_args()

    w1 = zeros((IN_DIM, H))
    b1 = zeros((1, H))
    w2 = zeros((H, CLASSES))
    b2 = zeros((1, CLASSES))
    load_weights(args.weights, [w1, b1, w2, b2])
    print(f"Loaded weights from {args.weights}")

    root = tk.Tk()
    DrawApp(root, w1, b1, w2, b2)
    root.mainloop()


if __name__ == '__main__':
    main()
