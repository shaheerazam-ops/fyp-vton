# FitLayer standalone virtual try-on widget

This is **only the widget**, without a sample e-commerce page or other pages. It opens directly at the root URL. The garment preview defaults to a sample image, and **Testing options** permits replacing it for local two-image API testing.

## Run locally

1. Start the FastAPI demo backend from your `Backend` folder:
   `python -m uvicorn api.local_demo:app --reload --port 8000`
2. In this frontend folder run:
   `npm install`
   `npm run dev`
3. Open `http://127.0.0.1:5173/`.

Vite proxies `/api/v1/try-on` to `http://127.0.0.1:8000/v1/try-on`. The frontend sends `person_photo`, `garment_photo`, `category`, `description`, and `seed` as multipart form data, and displays the returned image.

This is a **standalone widget testing view**, not yet a distributable embed SDK. Production retailer integration will need a separate initialization and garment-selection interface.
