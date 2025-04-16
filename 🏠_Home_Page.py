import streamlit as st
import os

st.set_page_config(layout="wide")
st.set_option('client.showErrorDetails', False)

st.markdown("""
    <style>
    div[data-testid="stSidebarCollapseButton"] {
        display: none;
    }
    a[data-testid="stSidebarNavLink"] {
    background-color: rgba(255,255,255,0.1)  !important
    }
    a[data-testid="stSidebarNavLink"][aria-current="page"] {
    background-color: #4CAF50;  /* Green color for active link */
    color: white;  /* White text */
    font-weight: bold;  /* Bold text */
}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('📖 Learn about the model in this [blog](https://medium.com/@ml_schmidt)!')
    "---"
    "[View the source code](https://github.com/MLSchmidtverse)"

st.title("📰🔍Fake News Detector")
st.write("🚨 This app helps you assess the likelihood that a news article is fake, using advanced machine learning techniques. With a few simple steps, you can check the trustworthiness of news content before sharing or believing it.")
st.write("")
st.write("Select one of the analysis options from the sidebar to begin your investigation:")
st.markdown("- **Text Input:** Type or paste the text of a news article, and watch the app analyze its authenticity based on a trained machine learning model.")
st.markdown("- **Upload:** Upload a text file, and let the app examine its content for potential misinformation, giving you an instant suspicion score.")
st.markdown("- **URL:** Analyze the content of a web page directly by entering its URL. The app checks the trustworthiness of online news sources in seconds.")
st.markdown("- **Fact-Checking Resources:** Curious about how to fact-check articles yourself? Explore tools and tips for verifying news with reliable sources.")
st.write("")
st.write("✨ **What's new?**")
st.write("The app is evolving to provide even more value:")
st.markdown("- **Sus Score per Paragraph:** If an article is suspicious, the app breaks it down into paragraphs and highlights the most suspicious parts, making it easier to spot potentially fake sections.")
st.markdown("- **Enhanced Upload Options:** The app supports more file formats for uploads, allowing analysis of a wider range of documents. Currently txt, pdf and docx files are supported.")
st.markdown("- **Currently optimized** for three major news websites (BBC News, Fox News, Al Jazeera), with plans to improve the web scraper to support even more sources, offering a wider range of news to analyze.")
st.write("")
st.write("🚀 **What's next?**")
st.write("There is a lot to do to tackle fake news - here are some ideas for future features:")
st.markdown("- **Fake News Tracking:** Currently, the app does not store any data, but one possible feature is to keep a track of fake news for specific authors or news sites. This would help identify patterns over time.")
st.markdown("- **Improved Short Text Analysis:** The current model is trained on articles with an average length of ~500 words. While it perfoms well on shorter texts of around 600 characters, future improvements will focus on even shorter texts, making it applicable to tweets and social media posts.")
st.markdown("- **Visual Content Verification:** Fake images and videos are an increasing concern. Plans are underway to incorporate visual analysis tools to assess the credibility of multimedia content, allowing images and videos to be verified alongside textual news.")