# 🌸 Image Classifier API with FastAPI

This is a simple API built using **FastAPI** that loads a pre-trained **Iris model** (`iris_model.joblib`) and predicts the class based on input features.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI app:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Open in browser:**
   - Go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to see the interactive Swagger UI!

---

## 📦 API Endpoints

- **GET /** → Returns a welcome message.
- **POST /predict** → Send a list of 4 features, returns the predicted Iris class.

Example Request Body for `/predict`:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```
---

## 💬 Notes
- The model is trained on the Iris dataset.
- Make sure `iris_model.joblib` is present in the same folder as `main.py`.
