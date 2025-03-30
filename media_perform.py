import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os  # Path Handling

# Load Model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = r"D:\social_media_perform_analysis\social_mediam_odel.joblib"
model = joblib.load(model_path)

# **Custom Styling**
st.markdown("""
    <style>
        body { background-color: #f8f9fa; }
        .big-title { text-align: center; color: black; font-size: 36px; font-weight: bold; }
        .sub-title { text-align: center; color: black; padding-bottom: 10px; font-size: 24px; }
        .upload-box { border: 3px dashed #ff4c4c; padding: 20px; text-align: center; border-radius: 10px; background-color: #222; color: #fff; font-size: 18px; }
        .stButton>button { font-size: 22px; padding: 12px 25px; border-radius: 10px; background: linear-gradient(45deg, #ff4c4c, #ff0080); color: white; border: none; width: 100%; }
        .stDownloadButton>button { background: linear-gradient(45deg, #4CAF50, #00FF00); color: white; border-radius: 10px; padding: 12px 25px; font-size: 18px; width: 100%; }
        .stRadio > div { display: flex; justify-content: center; gap: 20px; }
        .stRadio div[role=radio] { background-color: #444; padding: 12px 20px; border-radius: 10px; font-size: 18px; color: #fff; text-align: center; min-width: 200px; }
        .data-preview { border-radius: 10px; background: #222; padding: 20px; color: #fff; font-size: 18px; }
        .stTextInput>div>div>input { font-size: 20px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="big-title">Social Media Performance Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">AI-powered predictions for social media engagement</h3>', unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #555;'>", unsafe_allow_html=True)

# Select Input Type
option = st.radio("Select Input Method:", ["Upload CSV File", "Manual Input"])

# Required Features
req_features = ['Platform', 'Post_Type', 'Likes', 'Shares', 'Comments', 
                'Impressions', 'Shares_per_Impression', 'Engagement_per_Impression']

def clean_and_impute(df):
    """Handles missing, invalid, and out-of-range values."""
    df.replace(["error", "NA", "??", "none", "missing", ""], np.nan, inplace=True)

    # Convert numeric columns
    numeric_cols = ['Likes', 'Shares', 'Comments', 'Impressions', 'Shares_per_Impression', 'Engagement_per_Impression']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    valid_ranges = {
        "Likes": (1, 100000),
        "Shares": (1, 100000),
        "Comments": (1, 100000),
        "Impressions": (1, 100000),
        "Shares_per_Impression": (0.01, 100000),
        "Engagement_per_Impression": (0.01, 100000)
    }

    for col in valid_ranges:
        if df[col].isna().all():
            df[col].fillna((valid_ranges[col][0] + valid_ranges[col][1]) / 2, inplace=True)
        else:
            col_mean = df[col].mean(skipna=True)
            df[col] = df[col].apply(lambda x: col_mean if pd.isna(x) or x < valid_ranges[col][0] or x > valid_ranges[col][1] else x)

    for col in ['Platform', 'Post_Type']:
        if not df[col].mode().empty:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna("Unknown", inplace=True)

    # Fix spelling mistakes
    df['Platform'] = df['Platform'].replace({
        "Twittr": "Twitter",
        "Facebok": "Facebook",
        "Insta": "Instagram"
    })

    df['Post_Type'] = df['Post_Type'].replace({
        "Txx": "Text",
        "Img": "Image",
        "Vid": "Video"
    })

    # Replace "Unknown" with the most common category if available, else default
    if "Unknown" in df["Platform"].values:
        most_common_platform = df["Platform"].mode()[0] if not df["Platform"].mode().empty else "Twitter"
        df["Platform"] = df["Platform"].replace("Unknown", most_common_platform)

    if "Unknown" in df["Post_Type"].values:
        most_common_post_type = df["Post_Type"].mode()[0] if not df["Post_Type"].mode().empty else "Image"
        df["Post_Type"] = df["Post_Type"].replace("Unknown", most_common_post_type)

    # Encode categorical values into numbers
    df['Platform'] = df['Platform'].replace({"Facebook": 0, "Instagram": 1, "LinkedIn": 2, "Twitter": 3})
    df['Post_Type'] = df['Post_Type'].replace({"Image": 0, "Link": 1, "Text": 2, "Video": 3})

    return df

if option == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Raw Data Preview:**")
            st.dataframe(df.head())

            missing_features = [col for col in req_features if col not in df.columns]
            if missing_features:
                st.error(f"Missing Features: {missing_features}")
            else:
                df = clean_and_impute(df)
                
                # Ensure no NaN values before prediction
                if df.isna().sum().sum() > 0:
                    st.error("Data contains missing values after cleaning. Please check input data.")
                else:
                    predictions = model.predict(df[req_features].to_numpy())  
                    df["Predicted_Engagement_Rate"] = predictions  

                    st.success("**Predicted Results:**")
                    st.dataframe(df)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Predictions", csv, "predicted_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {str(e)}")

elif option == "Manual Input":
    st.subheader("Enter Values Manually:")

    # Categorical Inputs with Mapping
    platform_mapping = {"facebook": 0, "instagram": 1, "linkedin": 2, "twitter": 3}
    post_type_mapping = {"image": 0, "link": 1, "text": 2, "video": 3}
    Platform = st.selectbox("Platform", list(platform_mapping.keys()))
    Post_Type = st.selectbox("Post Type", list(post_type_mapping.keys()))

    Platform = platform_mapping[Platform]
    Post_Type = post_type_mapping[Post_Type]

    Likes = st.number_input("Likes", min_value=0, step=5, value=230)
    Shares = st.number_input("Shares", min_value=0, step=5, value=40)
    Comments = st.number_input("Comments", min_value=0, step=5, value=35)
    Impressions = st.number_input("Impressions", min_value=0, step=10, value=5000)
    Shares_per_Impression = st.number_input("Shares per Impression", min_value=0.0010, step=0.5, value=0.008)
    Engagement_per_Impression = st.number_input("Engagement per Impression", min_value=0.001460, step=0.5, value=0.061)

    if st.button("Predict"):
        input_data = np.array([[Platform, Post_Type, Likes, Shares, Comments, Impressions, Shares_per_Impression, Engagement_per_Impression]])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Engagement Rate: **{prediction:.2f}**")