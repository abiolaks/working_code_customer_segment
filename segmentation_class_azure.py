import streamlit as st
from openai import AzureOpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.graph_objects as go


class CustomerSegmentationApp_1:
    def __init__(self):
        self.data = None
        self.clustered_data = None
        self.client = self.get_azure_openai_client()

    def get_azure_openai_client(self):
        """Initialize and return the Azure OpenAI API client."""
        api_key = st.secrets["AZURE_OPENAI_API_KEY"]
        azure_endpoint = st.secrets["ENDPOINT_URL"]
        deployment_name = st.secrets["DEPLOYMENT_NAME"]
        api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
        if not api_key or not azure_endpoint or not deployment_name:
            st.error(
                "Azure OpenAI configuration missing. Please set your API key, endpoint, and deployment name."
            )
            return None
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=deployment_name,
        )

    def load_data(self):
        """Handle file upload and load customer data."""
        upload_file = st.file_uploader(
            "upload your telecom customer data (default: CSV)", type="csv"
        )
        if upload_file:
            self.data = pd.read_csv(upload_file)
            st.write("Data Preview")
            st.write(self.data.head())
        else:
            st.warning("Please upload a csv file.")
        return self.data

    def preprocess_data(self):
        """Preprocess the data: scaling, encoding, and feature engineering."""
        if self.data is not None:
            categorical_features = ["Gender", "Region"]
            self.data = pd.get_dummies(
                self.data, columns=categorical_features, drop_first=True
            )
            self.data.fillna(self.data.median(), inplace=True)
            self.data["Recency"] = 30 - self.data["LastPurchaseDays"]
            self.data["Frequency"] = self.data["CallsMade"] / 30
            self.data["Monetary"] = self.data["MonthlySpending"]
            scaler = StandardScaler()
            features = ["Recency", "Frequency", "Monetary", "DataUsageGB"]
            self.data[features] = scaler.fit_transform(self.data[features])
            return self.data

    def cluster_data(self, num_clusters):
        """Perform customer segmentation using KMeans clustering."""
        if self.data is not None:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            self.data["Cluster"] = kmeans.fit_predict(
                self.data[["Recency", "Frequency", "Monetary", "DataUsageGB"]]
            )
            self.clustered_data = self.data
            return self.clustered_data

    def plot_clusters(self):
        """Visualize the clusters in 3D using Plotly."""
        if self.clustered_data is not None:
            fig = go.Figure()
            for cluster_id in sorted(self.clustered_data["Cluster"].unique()):
                cluster_data = self.clustered_data[
                    self.clustered_data["Cluster"] == cluster_id
                ]
                fig.add_trace(
                    go.Scatter3d(
                        x=cluster_data["Recency"],
                        y=cluster_data["Frequency"],
                        z=cluster_data["Monetary"],
                        mode="markers",
                        marker=dict(size=5, opacity=0.8),
                        name=f"Cluster {cluster_id}",
                    )
                )
            fig.update_layout(
                title="3D Customer Segmentation",
                scene=dict(
                    xaxis_title="Recency",
                    yaxis_title="Frequency",
                    zaxis_title="Monetary",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                legend=dict(x=0.1, y=0.9),
            )
            return fig

    def generate_cluster_insights(self, data):
        self.data = data
        """Generate insights for each cluster using Azure OpenAI."""
        insights = []
        if self.client is None:
            st.error(
                "Azure OpenAI client is not initialized. Please check your configuration."
            )
            return insights
        if self.data is None:
            st.error("Clustered data is not available. Please run clustering first.")
            return insights
        system_prompt = """
            You are a Telecommunication Customer Insights Analyst. You are tasked with analyzing Customer Clusters
            for actionable insights.
            
            Give a concise and detailed response with the specified output format below in an easy-to-understand manner
            for the Marketing, Sales Team to act on.
            
            Output in Markdown
            1. Demographic Insights
            2. Customer Behaviour Analysis
            3. Tailor Marketing Strategies
            4. Product and Pricing Strategies
            
            Strictly stick to the output and format it in Markdown for each cluster.
            in your response do not give values with negative numerical values.use absolute values.instead of saying -0.5 say 0.5
            """
        cluster_summary = self.clustered_data.groupby("Cluster").mean().reset_index()
        deployment_name = st.secrets.get("DEPLOYMENT_NAME")
        for cluster_id in cluster_summary["Cluster"]:
            try:
                cluster_data = cluster_summary.loc[
                    cluster_summary["Cluster"] == cluster_id
                ]
                prompt = f"Analyze the following customer data for Cluster {cluster_id}: {cluster_data.to_dict()}"
                response = self.client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    top_p=1,
                    max_tokens=600,
                )
                insights.append(response.choices[0].message.content)
            except Exception as e:
                st.error(
                    f"Error generating insights for Cluster {cluster_id}: {str(e)}"
                )
                insights.append(f"Error generating insights for Cluster {cluster_id}.")
        return insights

    def get_clustered_data(self):
        """Return the clustered data for download."""
        return self.clustered_data
