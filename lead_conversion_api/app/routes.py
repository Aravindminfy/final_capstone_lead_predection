from flask import request, render_template, redirect, url_for
import pandas as pd

def register_routes(app):

    @app.route('/')
    def home():
        return render_template('upload.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return "❌ No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "❌ No file selected", 400

        try:
            df = pd.read_csv(file)

            # Dummy logic: predict 1 if TotalVisits > 3
            predictions = [1 if visits > 3 else 0 for visits in df['TotalVisits']]
            result = list(zip(df['TotalVisits'], predictions))

            return f'''
                <h2>Prediction Results</h2>
                <ul>
                    {''.join(f'<li>TotalVisits: {v}, Prediction: {p}</li>' for v, p in result)}
                </ul>
                <a href="/">🔙 Go back</a>
            '''
        except Exception as e:
            return f"❌ Error reading file: {e}", 500
