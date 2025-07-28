# Multimodal-ML--Housing-Price-Prediction-Using-Images

🏠 Housing Price Prediction Using Images + Tabular Data
📌 Objective
Build a multimodal machine learning model that predicts housing prices using both:

Tabular (structured) data (e.g., number of rooms, square footage, location)

Images of the houses (e.g., exterior/interior photos)

This project showcases how to combine image features extracted via CNNs with structured data to improve regression performance.

📂 Dataset
Use a dataset containing:

Tabular features such as price, size, location, bedrooms, etc.

Image data (associated with each house ID)

Sources:

📊 Tabular Data: Kaggle datasets, Zillow, or similar sources

🖼 Image Data: Custom scraped images or open image datasets

⚙️ Technologies Used
Python

Pandas, NumPy

Scikit-learn

TensorFlow / PyTorch (for CNN image feature extraction)

Matplotlib / Seaborn

Joblib (optional for model saving)

🧠 Steps and Workflow
1. Data Preprocessing
Clean and normalize tabular features

Resize and standardize image dimensions

Match images with corresponding tabular entries (by unique IDs)

2. Feature Engineering
Use a Convolutional Neural Network (CNN) to extract image embeddings (e.g., using pre-trained ResNet50, VGG16)

Flatten CNN output and combine with tabular data

3. Model Training
Concatenate CNN image features and tabular features

Use models like:

Fully connected neural network

Gradient Boosting Regressor

Train using appropriate train-test split

4. Evaluation Metrics
Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Optionally: R² score

5. Visualization
Actual vs Predicted prices

Error distribution

🛠 Skills Gained
✅ Multimodal Machine Learning (image + tabular)
✅ CNNs for Image Feature Extraction
✅ Feature Fusion Techniques
✅ Regression Modeling & Evaluation
✅ Deep Learning Integration with Traditional ML