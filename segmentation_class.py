import streamlit as st
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from openai import AzureOpenAI

# import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


class CustomerSegmentationApp_1:
    def __init__(self):
        self.data = None
        self.clustered_data = None
        self.client = self.get_openai_client()

    def get_openai_client(self):
        """_summary_: Initialize and return the OpenAI API client.
        _return_: OpenAI API client
        """
        api_key = st.secrets["secrets"]["OPENAI_API_KEY"]
        if not api_key:
            st.error("No OpenAI key found, Please set your API key. ")
            return None
        return OpenAI(api_key=api_key)

    def load_data(self):
        """_summary_:Handle file upload and load customer data.
        _return_: the loaded data
        """
        upload_file = st.file_uploader(
            "upload your telecom customer data (default: CSV)", type="csv"
        )
        if upload_file:
            self.data = pd.read_csv(
                upload_file
            )  # convert csv to dataframe and save in data
            st.write("Data Preview")
            st.write(self.data.head())
        else:
            st.warning("Please upload a csv file.")

        return self.data

    def preprocess_data(self):
        """Preprocess the data: scaling, encoding, and feature engineering."""
        if self.data is not None:
            # Handle categorical features
            categorical_features = ["Gender", "Region"]
            self.data = pd.get_dummies(
                self.data, columns=categorical_features, drop_first=True
            )

            # Fill missing values
            self.data.fillna(self.data.median(), inplace=True)

            # Feature engineering
            self.data["Recency"] = 30 - self.data["LastPurchaseDays"]
            self.data["Frequency"] = self.data["CallsMade"] / 30
            self.data["Monetary"] = self.data["MonthlySpending"]

            # Scale numerical features
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
            # Create a 3D scatter plot
            fig = go.Figure()

            # Add a trace for each cluster
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

            # Update layout for better visualization
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
        """Generate insights for each cluster using OpenAI."""
        insights = []  # Initialize an empty list to store insights

        # Check if the OpenAI client is available
        if self.client is None:
            st.error("OpenAI client is not initialized. Please check your API key.")
            return insights  # Return an empty list

        # Check if clustered data is available
        if self.data is None:
            st.error("Clustered data is not available. Please run clustering first.")
            return insights  # Return an empty list

        # System prompt for OpenAI
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

        # Generate cluster summary statistics
        cluster_summary = self.clustered_data.groupby("Cluster").mean().reset_index()

        # Generate insights for each cluster
        for cluster_id in cluster_summary["Cluster"]:
            try:
                # Prepare the prompt for OpenAI
                cluster_data = cluster_summary.loc[
                    cluster_summary["Cluster"] == cluster_id
                ]
                prompt = f"Analyze the following customer data for Cluster {cluster_id}: {cluster_data.to_dict()}"

                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    top_p=1,
                    max_tokens=600,
                )

                # Append the generated insight to the list
                insights.append(response.choices[0].message.content)

            except Exception as e:
                # Log any errors that occur during the API call
                st.error(
                    f"Error generating insights for Cluster {cluster_id}: {str(e)}"
                )
                insights.append(f"Error generating insights for Cluster {cluster_id}.")

        return insights  # Return the list of insights

    def get_clustered_data(self):
        """Return the clustered data for download."""
        return self.clustered_data
