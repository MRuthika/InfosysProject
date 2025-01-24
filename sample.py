import streamlit as st
import torch
import numpy as np
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    pipeline
)
from sklearn.cluster import KMeans
from collections import Counter 
import tempfile
import os
import openai

# OpenAI API Key (replace with your actual key)
OPENAI_API_KEY = 'your_openai_api_key_here'
openai.api_key = OPENAI_API_KEY

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

class AdvancedImageProcessor:
    def __init__(self, 
                 caption_model="Salesforce/blip-image-captioning-large", 
                 sentiment_model="distilbert-base-uncased-finetuned-sst-2-english"):
        # Caption model
        self.caption_processor = BlipProcessor.from_pretrained(caption_model)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_model)
        
        # Sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        
        self.pil_image = None
        self.cv_image = None

    def load_image(self, image_path):
        """Load image and convert to PIL and OpenCV formats."""
        self.pil_image = Image.open(image_path).convert("RGB")
        self.cv_image = cv2.cvtColor(np.array(self.pil_image), cv2.COLOR_RGB2BGR)
        return self.pil_image, self.cv_image

    def generate_image_context(self):
        """Generate detailed contextual description of the image."""
        inputs = self.caption_processor(images=self.pil_image, return_tensors="pt")
        output = self.caption_model.generate(**inputs, max_length=150)
        return self.caption_processor.decode(output[0], skip_special_tokens=True)

    def generate_enhanced_text(self, context):
        """Generate enhanced description using OpenAI's API."""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a detailed image description assistant."},
                    {"role": "user", "content": f"Provide a vivid, concise 3-4 sentence description that goes beyond the basic context. Original context: {context}"}
                ],
                max_tokens=200,
                temperature=0.7
            )
            enhanced_text = response.choices[0].message.content.strip()
            return enhanced_text
        except Exception as e:
            return f"Error generating enhanced text: {str(e)}"

    def sentiment_analysis(self, text):
        """Simplify sentiment analysis to return only one primary sentiment."""
        sentiment = self.sentiment_analyzer(text)[0]
        return pd.DataFrame([{
            'Sentiment': sentiment['label'],
            'Confidence': round(sentiment['score'] * 100, 2)
        }])

    def analyze_colors(self, k=5):
        """Analyze dominant colors in the image."""
        image_data = self.cv_image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(image_data)
        
        dominant_colors = kmeans.cluster_centers_.astype(int)
        color_counts = Counter(kmeans.labels_)
        
        # Color histogram
        plt.figure(figsize=(10, 5))
        plt.title("Color Distribution")
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            histogram = cv2.calcHist([self.cv_image], [i], None, [256], [0, 256])
            plt.plot(histogram, color=color)
        color_hist_fig = plt.gcf()
        
        # Pie chart of dominant colors
        plt.figure(figsize=(8, 8))
        plt.title("Dominant Colors")
        percentages = [count / sum(color_counts.values()) * 100 for count in color_counts.values()]
        color_normalized = [c/255.0 for c in dominant_colors]
        plt.pie(percentages, colors=color_normalized, 
                labels=[f'Color {i+1}' for i in range(len(percentages))], 
                autopct='%1.1f%%')
        color_pie_fig = plt.gcf()
        
        return color_hist_fig, color_pie_fig

def main():
    st.set_page_config(layout="wide", page_title="Image Analysis")
    st.title("Real txt to Alt txt and Enhanced visualization")
    
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        temp_path = save_uploaded_file(uploaded_file)
        processor = AdvancedImageProcessor()
        original_image, _ = processor.load_image(temp_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("Analyze Image"):
                # Image Context
                st.subheader("Image Context")
                image_context = processor.generate_image_context()
                st.write(image_context)
                
                # Enhanced Text
                st.subheader("Enhanced Context")
                enhanced_text = processor.generate_enhanced_text(image_context)
                st.write(enhanced_text)
                
                # Color Histogram and Pie Chart
                st.subheader("Color Analysis")
                color_hist, color_pie = processor.analyze_colors()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.pyplot(color_hist)
                with col_b:
                    st.pyplot(color_pie)
                
                # Sentiment Analysis
                st.subheader("Sentiment Analysis")
                sentiment_df = processor.sentiment_analysis(enhanced_text)
                st.dataframe(sentiment_df)

    else:
        st.info("Please upload an image to begin analysis.")

if __name__ == "__main__":
    main()