import streamlit as st

from segmentation_class_azure import CustomerSegmentationApp_1

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #2E86C1;
    }
    .stSubheader {
        color: #1A5276;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    app = CustomerSegmentationApp_1()

    # Title and Header
    st.title("📊 Customer Value Management")
    st.header("Customer Segmentation and Insights UseCase")
    st.markdown("""
    This application leverages advanced clustering techniques and LLMs to provide actionable insights into customer segments for telcos.
    """)
    st.divider()

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Settings")
        num_clusters = st.slider(
            "Select Number of Clusters", min_value=2, max_value=10, value=3
        )
        st.markdown("Adjust the number of clusters for segmentation.")

    # Load and preprocess data
    st.subheader("🔧 Data Loading and Preprocessing")
    with st.spinner("Loading and preprocessing data..."):
        app.load_data()
        app.preprocess_data()
    st.success("Data loaded and preprocessed successfully!")
    st.divider()

    # Clustering and Analysis
    st.subheader("📈 Clustering and Customer Segment Analysis")
    if st.button("Run Clustering"):
        with st.spinner("Running clustering algorithm..."):
            app.cluster_data(num_clusters)
            st.success("Clustering completed!")
            st.plotly_chart(app.plot_clusters())

    st.divider()

    # Cluster Insights
    data = app.cluster_data(num_clusters)
    st.subheader("🔍 Cluster Insights")
    if st.button("Generate Insights"):
        with st.spinner("Generating insights using LLMs..."):
            insights = app.generate_cluster_insights(data)
            st.success("Insights generated successfully!")
            for i, insight in enumerate(insights):
                with st.expander(f"Cluster {i+1} Insights"):
                    st.markdown(insight)

    st.divider()

    # Optional: Raw Data Preview
    if hasattr(app, "get_raw_data") and st.checkbox("Show Raw Data"):
        st.subheader("📄 Raw Data Preview")
        st.dataframe(app.get_raw_data())


if __name__ == "__main__":
    main()
