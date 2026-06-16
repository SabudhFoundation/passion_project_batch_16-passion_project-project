import io
import base64
import requests
import gradio as gr
from PIL import Image as PILImage

API_BASE = "http://127.0.0.1:8000"

VALID_METHODS = [
    "lbp_svm", "fft_svm", "combined_svm",
    "glcm_mlp", "lbp_mlp", "fft_mlp", "combined_mlp",
    "resnet50", "inceptionv3", "efficientnet",
    "dual_channel_inception", "dual_channel_resnet",
]


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _img_html(pil_img: PILImage.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        '<div style="width:100%;border-radius:10px;overflow:hidden;'
        'background:#0f172a;line-height:0;">'
        f'<img src="data:image/jpeg;base64,{b64}" '
        'style="width:100%;height:340px;object-fit:contain;display:block;"/>'
        "</div>"
    )


def _img_placeholder() -> str:
    return (
        '<div style="width:100%;height:340px;border-radius:10px;'
        'border:2px dashed #334155;display:flex;flex-direction:column;'
        'align-items:center;justify-content:center;gap:10px;'
        'color:#475569;background:#0f172a;">'
        '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" '
        'stroke="#475569" stroke-width="1.5" stroke-linecap="round">'
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<circle cx="8.5" cy="8.5" r="1.5"/>'
        '<polyline points="21,15 16,10 5,21"/>'
        "</svg>"
        '<span style="font-size:0.9rem;">No image uploaded yet</span>'
        "</div>"
    )


def _results_placeholder(msg: str = "") -> str:
    text = msg or "Upload an image and click <b>Classify</b>"
    return (
        '<div style="height:220px;border-radius:10px;border:2px dashed #334155;'
        'display:flex;align-items:center;justify-content:center;'
        'color:#475569;font-size:0.95rem;background:#0f172a;">'
        f"{text}</div>"
    )


def _results_html(label: str, confidence: str, method: str, accuracy: str) -> str:
    is_fake   = label.lower() == "fake"
    lbl_color = "#ef4444" if is_fake else "#22c55e"
    lbl_bg    = "#450a0a" if is_fake else "#052e16"
    lbl_bdr   = "#991b1b" if is_fake else "#166534"
    icon      = "&#x2717;" if is_fake else "&#x2713;"

    def row(key, val, last=False):
        border = "" if last else "border-bottom:1px solid #1e293b;"
        return (
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;padding:14px 0;{border}">'
            f'<span style="font-size:0.78rem;font-weight:700;color:#64748b;'
            f'text-transform:uppercase;letter-spacing:0.07em;">{key}</span>'
            f'<span style="font-size:1rem;font-weight:600;color:#e2e8f0;">{val}</span>'
            f"</div>"
        )

    return (
        '<div style="display:flex;flex-direction:column;gap:16px;">'

        # Label badge
        f'<div style="background:{lbl_bg};border:2px solid {lbl_bdr};'
        f'border-radius:10px;padding:30px;text-align:center;">'
        f'<div style="font-size:3rem;font-weight:900;color:{lbl_color};'
        f'letter-spacing:0.12em;">{icon}&nbsp;{label.upper()}</div>'
        f"</div>"

        # Stats card
        '<div style="background:#0f172a;border:1px solid #1e293b;'
        'border-radius:10px;padding:4px 20px;">'
        f"{row('Confidence', confidence)}"
        f"{row('Method', method)}"
        f"{row('Reported accuracy', accuracy, last=True)}"
        "</div>"

        "</div>"
    )


def _error_html(msg: str) -> str:
    return (
        '<div style="border-radius:10px;border:2px solid #92400e;'
        'background:#451a03;padding:20px;text-align:center;'
        'color:#fb923c;font-size:0.9rem;">&#9888;&nbsp;&nbsp;'
        f"{msg}</div>"
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def on_upload(file_path: str):
    """Called when user picks a file. Returns (image_html, results_html, pil_img)."""
    if file_path is None:
        return _img_placeholder(), _results_placeholder(), None
    try:
        img = PILImage.open(file_path).convert("RGB")
        return _img_html(img), _results_placeholder(), img
    except Exception as exc:
        return _img_placeholder(), _error_html(f"Could not open image: {exc}"), None


def classify(pil_image, method: str):
    """POST the stored PIL image to the FastAPI endpoint."""
    if pil_image is None:
        return _results_placeholder("Please upload an image first.")
    if not method:
        return _results_placeholder("Please select a detection method.")

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=92)
    buf.seek(0)

    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            data={"method": method},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return _error_html(
            f"Cannot reach API at <code>{API_BASE}</code>. "
            "Start it with: <code>python -m uvicorn src.inference.api:app</code>"
        )
    except requests.exceptions.Timeout:
        return _error_html("Request timed out — model may still be loading.")

    if resp.status_code == 200:
        d = resp.json()
        return _results_html(
            label      = d["label"],
            confidence = f"{d['confidence'] * 100:.1f}%",
            method     = d["method"],
            accuracy   = f"{d['reported_accuracy']}%",
        )
    elif resp.status_code == 503:
        return _error_html(
            f"<b>{method}</b> — weights file not found. "
            "Train or download the model and place it in <code>models/</code>."
        )
    else:
        detail = resp.json().get("detail", resp.text)
        return _error_html(f"API error {resp.status_code}: {detail}")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
.panel { background:#111827; border:1px solid #1f2937;
         border-radius:14px; padding:24px; }
.lbl   { font-size:0.72rem !important; font-weight:700 !important;
         letter-spacing:0.1em !important; text-transform:uppercase !important;
         color:#6b7280 !important; margin:0 0 10px !important; }
footer { display:none !important; }
"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Deepfake Image Detector", theme=gr.themes.Base(), css=CSS) as demo:

    # Hidden state — holds the PIL image between upload and classify
    pil_state = gr.State(value=None)

    gr.HTML("""
    <div style="padding:8px 0 20px;">
      <h1 style="margin:0;font-size:1.9rem;font-weight:800;color:#f9fafb;">
        Deepfake Image Detector
      </h1>
      <p style="margin:6px 0 0;color:#9ca3af;font-size:0.95rem;">
        Upload a face image, choose a detection pipeline,
        then click <strong style="color:#e5e7eb;">Classify</strong>.
      </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left panel ──────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="panel"):

            gr.HTML('<p class="lbl">Image preview</p>')
            image_display = gr.HTML(value=_img_placeholder())

            # Single upload button — no image preview widget, no duplicates
            upload_btn = gr.UploadButton(
                "Click to upload / drag & drop",
                file_types=["image"],
                variant="secondary",
                size="sm",
            )

            method_dropdown = gr.Dropdown(
                choices=VALID_METHODS,
                value="fft_svm",
                label="Detection method",
            )
            classify_btn = gr.Button("Classify", variant="primary", size="lg")

        # ── Right panel ─────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="panel"):
            gr.HTML('<p class="lbl">Prediction</p>')
            results_panel = gr.HTML(value=_results_placeholder())

    # ── Events ──────────────────────────────────────────────────────────────
    upload_btn.upload(
        fn=on_upload,
        inputs=[upload_btn],
        outputs=[image_display, results_panel, pil_state],
    )

    classify_btn.click(
        fn=classify,
        inputs=[pil_state, method_dropdown],
        outputs=[results_panel],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)
