from django.shortcuts import render,redirect,HttpResponse,get_object_or_404
from django.views import View
import pandas as pd
import plotly.express as px
from dashboard.models import UploadedFile
import json
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from plotly.offline import plot
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from django.contrib.auth.mixins import LoginRequiredMixin
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



class DataAnalysisView(LoginRequiredMixin,View):
    template_name = 'main/analysis.html'
    template_cl = None

    target_column_keywords = [
            'target', 'label', 'outcome', 'prediction', 'score', 'result', 'response',
            'class', 'category', 'status', 'flag', 'value', 'rating', 'grade', 'level',
            'quantity', 'amount', 'total', 'sum', 'average', 'ratio', 'percentage',
            'frequency', 'count', 'occurrence', 'frequency', 'probability', 'churn',
            'survival', 'conversion', 'income', 'revenue', 'profit', 'loss', 'expense',
            'price', 'cost', 'discount', 'growth', 'trend', 'change', 'delta', 'variance',
            'deviation', 'error', 'residual',
            'age', 'sex', 'gender', 'height', 'weight', 'income', 'salary', 'education',
            'country', 'city', 'region', 'state', 'zip', 'address', 'location', 'latitude', 'longitude',
            'customer', 'client', 'user', 'member', 'employee', 'patient', 'student', 'citizen', 'resident',
            'purchase', 'sale', 'transaction', 'order', 'invoice', 'payment', 'delivery', 'shipment', 'refund',
            'product', 'item', 'service', 'inventory', 'stock', 'quantity', 'price', 'cost',
        ]

    def get(self, request, file_id):
        uploaded_file = get_object_or_404(UploadedFile, id=file_id)
        data = pd.read_csv(uploaded_file.file)
        # try:
        context = self.analyze_data(data, uploaded_file)
        # except Exception as e:
        #     context = {
        #         'error_message' :f"Unable to analyze the data\n {str(e)}",
        #     }
        return render(request, self.template_name, context)

    def analyze_data(self, data, uploaded_file):
        context = {}

        # Feature: Show data summary
        data_summary = data.describe().transpose().to_html(classes='table table-hover table-sm table-responsive nvs-table-data-ds')
        context['data_summary'] = data_summary

        # Feature: Identify and Select Columns
        useful_columns = self.get_useful_columns(data)
        y_column = self.auto_identify_target_column(data, useful_columns)
        self.y_column = y_column
        top_columns = self.select_top_columns(useful_columns, data, y_column)
        if not top_columns or not y_column:
            context['error_message'] = "Unable to analyze the data"
            return context

        # Feature: Create Plots
        plots_html = self.create_plots(data, top_columns, y_column)
        context['plots_html'] = plots_html

        # Feature: Data Report
        report = f"Number of Rows: {len(data)}\n\nData Head:\n{data.head()}"
        context['report'] = report

        # Feature: Forecast Data
        forecast_data = self.forecast_data(data, y_column)
        context['forecast_data'] = forecast_data

        # Feature: Trend Analysis
        trend_data, trend_line = self.linear_regression(data, top_columns[0], y_column)
        context['trend_data'] = trend_data
        context['trend_line'] = trend_line

        # Feature: Anomaly Detection
        anomalies_s, anomalies = self.detect_anomalies(data, top_columns[0], y_column)
        context['anomalies_s'] = anomalies_s
        context['anomalies'] = anomalies

        # Additional features can be added here

        context.update({
            'uploaded_file': uploaded_file,
            'top_columns': top_columns,
            'y_column': y_column,
        })

        additional_visualizations = self.additional_visualizations(data)
        context.update(additional_visualizations)

        return context

    def get_useful_columns(self, data):
        numerical_columns = data.select_dtypes(include=['number']).columns
        datetime_columns = data.select_dtypes(include=['datetime']).columns
        object_columns = data.select_dtypes(include=['object']).columns

        # Choose numerical columns if available, else use datetime or object columns
        if not numerical_columns.empty or numerical_columns.length > 2:
            return numerical_columns
        elif not datetime_columns.empty:
            return datetime_columns
        else:
            return object_columns

    def select_top_columns(self, columns, data, y_column):
        top_columns = []

        for column in columns:
            if column == y_column:
                continue  # Skip selecting y_column again in top columns

            if pd.api.types.is_numeric_dtype(data[column]):
                # Logic for numerical columns, for example, select the top columns with highest mean
                top_columns.append(column)
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                # Logic for datetime columns, for example, select the top columns with earliest date
                top_columns.append(column)
            elif pd.api.types.is_object_dtype(data[column]):
                # Logic for object columns, for example, select the top columns with highest frequency
                top_columns.append(column)

        return top_columns[:8]

    def auto_identify_target_column(self, data, columns):

        # Check if any of the target_column_keywords is in the column names
        for column in data.columns:
            if any(keyword in column.lower() for keyword in self.target_column_keywords):
                return column

        # If none of the keywords is found, and the data contains numeric columns,
        # return the last numeric column as a fallback
        numeric_columns = data.select_dtypes(include=['number']).columns
        if not numeric_columns.empty:
            return numeric_columns[-1]

        # If there are no numeric columns, try to identify based on the specified columns
        for column in columns:
            if column in data.select_dtypes(include=['number']).columns:
                # Logic for numerical columns, for example, select the column with the highest mean
                return data[columns].mean().idxmax()
            elif column in data.select_dtypes(include=['datetime']).columns:
                # Logic for datetime columns, for example, select the column with the earliest date
                return data[columns].min().idxmin()
            elif column in data.select_dtypes(include=['object']).columns:
                # Logic for object columns, for example, select the column with the highest frequency
                return data[column].mode()[0]

        # If no target column is identified, return None
        return None
    
    def create_plots(self, data, columns, y_column):
        plots_html = []

        if y_column not in data.columns:
            return plots_html

        for column in columns:
            num_unique_values = data[column].nunique()

            if num_unique_values == 1:
                # Use a pie chart for a single unique value in the column
                fig = px.pie(data, names=column, title=f'Pie Chart: {column} Distribution', template=self.template_cl)
                self.add_unique_items_annotation(fig, data[column].nunique())
                plots_html.append(fig.to_html(full_html=False))
            elif num_unique_values == 2:
                # Use a bar plot for columns with two unique values
                fig = px.bar(data, x=column, y=y_column, color=column, title=f'Bar Plot: {column} vs {y_column} by {column}', template=self.template_cl)
                self.add_unique_items_annotation(fig, data[column].nunique())
                plots_html.append(fig.to_html(full_html=False))
            elif pd.api.types.is_numeric_dtype(data[column]):
                # Use a box plot for numerical data
                fig = px.box(data, x=column, y=y_column, points='all', title=f'Box Plot: {column} vs {y_column}', template=self.template_cl)
                self.add_unique_items_annotation(fig, data[column].nunique())
                plots_html.append(fig.to_html(full_html=False))
            elif pd.api.types.is_object_dtype(data[column]):
                # Use a bar plot for object columns
                fig = px.bar(data, x=column, y=y_column, title=f'Bar Plot: {column} vs {y_column}', template=self.template_cl)
                self.add_unique_items_annotation(fig, data[column].nunique())
                plots_html.append(fig.to_html(full_html=False))
            else:
                fig = px.scatter_matrix(data, dimensions=[column, y_column], color=column,
                                        title=f'Scatter Matrix: {column} vs {y_column}', template=self.template_cl)
                self.add_unique_items_annotation(fig, data[column].nunique())
                plots_html.append(fig.to_html(full_html=False))
                pass

        return plots_html
    
    def forecast_data(self, data, y_column):
        try:
            # Example: Using Exponential Smoothing for forecasting
            forecast_model = ExponentialSmoothing(data[y_column], seasonal='add', seasonal_periods=12)
            forecast_result = forecast_model.fit()
            forecast_data = forecast_result.forecast(12)  # Forecasting the next 12 periods
            return forecast_data.tolist()  # Convert to list for easy integration into template
        except Exception as e:
            # Handle any errors that may occur during forecasting
            return None

    def linear_regression(self, data, x_column, y_column):
        try:
            # Example: Using Linear Regression for trend analysis
            model = LinearRegression()
            X = data[[x_column]]
            y = data[y_column]
            model.fit(X, y)
            trend_line = pd.DataFrame({x_column: data[x_column], 'Trend': model.predict(X)})
            # Generate scatter plot using Plotly Express
            scatter_fig = px.scatter(data, x=x_column, y=y_column, title=f'Scatter Plot: {x_column} vs {y_column}', template=self.template_cl)
            trend_fig = px.line(trend_line, x=x_column, y='Trend', title=f'Linear Regression Trend: {x_column} vs {y_column}', template=self.template_cl)

            scatter_html = plot(scatter_fig, output_type='div', include_plotlyjs=False)
            trend_html = plot(trend_fig, output_type='div', include_plotlyjs=False)

            return scatter_html, trend_html
        
        except Exception as e:
            # Handle any errors that may occur during linear regression
            return None, None

    def detect_anomalies(self, data, x_column, y_column):
        try:
            # Example: Using Isolation Forest for anomaly detection
            clf = IsolationForest(contamination=0.05)  # Adjust the contamination parameter as needed
            X = data[[x_column, y_column]]
            data['Anomaly'] = clf.fit_predict(X)
            anomalies = data[data['Anomaly'] == -1]


            scatter_fig = px.scatter(data, x=x_column, y=y_column, title=f'Scatter Plot: {x_column} vs {y_column}', template=self.template_cl)
            anomalies_fig = px.scatter(anomalies, x=x_column, y=y_column, title=f'Anomalies: {x_column} vs {y_column}', template=self.template_cl)

            scatter_html = plot(scatter_fig, output_type='div', include_plotlyjs=False)
            anomalies_html = plot(anomalies_fig, output_type='div', include_plotlyjs=False)

            return scatter_html, anomalies_html
        except Exception as e:
            scatter_html = anomalies_html = None
            # Handle any errors that may occur during anomaly detection
            return scatter_html,anomalies_html
        

    def add_unique_items_annotation(self,fig, num_unique_items):
        fig.update_layout(
            annotations=[
                dict(
                    text=f"Unique Items: {num_unique_items}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.95, y=0.05
                ),
            ]
        )

    def additional_visualizations(self, data):
        selected_columns = self.select_top_columns22(data.columns, data)
        data = data[:10]
        # Histograms and Distribution Plots
        histograms = self.generate_histograms(data, selected_columns)

        # Correlation Matrix
        correlation_matrix = self.generate_correlation_matrix(data, selected_columns)

        # Box Plots
        box_plots = self.generate_box_plots(data, selected_columns)

        # Heatmap
        heatmap = self.generate_heatmap(data, selected_columns)

        # Pair Plots
        pair_plots = self.generate_pair_plots(data, selected_columns)

        # Principal Component Analysis (PCA)
        pca_plot = self.generate_pca_plot(data, selected_columns)

        context = {
            'histograms': histograms,
            'correlation_matrix': correlation_matrix,
            'box_plots': box_plots,
            'heatmap': heatmap,
            'pair_plots': pair_plots,
            'pca_plot': pca_plot,
        }

        return context

    def generate_histograms(self, data, columns):
        histograms_html = []

        for column in columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                # For numeric columns, create histograms with color
                hist_fig = px.histogram(data, x=column, title=f'Histogram: {column}', color=column, template=self.template_cl)
                histograms_html.append(hist_fig.to_html(full_html=False))

        return histograms_html

    def generate_correlation_matrix(self, data, columns):
        # Create a correlation matrix heatmap
        corr_matrix = data.select_dtypes(include=['number']).corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.index,
                                        colorscale='Viridis'))
        fig.update_layout(title='Correlation Matrix')
        correlation_matrix_html = fig.to_html(full_html=False)
        return correlation_matrix_html

    def generate_box_plots(self, data, columns):
        box_plots_html = []

        for column in columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                # For numeric columns, create box plots with color
                box_fig = px.box(data, y=column, title=f'Box Plot: {column}', color=column, template=self.template_cl)
                box_plots_html.append(box_fig.to_html(full_html=False))

        return box_plots_html

    def generate_heatmap(self, data, columns):
        # Generate a heatmap for the entire dataset
        heatmap_fig = px.imshow(data[columns], aspect='auto', color_continuous_scale='viridis', title='Heatmap', template=self.template_cl)
        heatmap_html = heatmap_fig.to_html(full_html=False)

        return heatmap_html

    def generate_pair_plots(self, data, columns):
        try:
            # Generate pair plots for numeric columns with color
            pair_plot_fig = px.scatter_matrix(data[columns], dimensions=columns, template=self.template_cl)
            pair_plot_html = pair_plot_fig.to_html(full_html=False)

            return pair_plot_html
        except Exception as e:
            return None

    def generate_pca_plot(self, data, columns):
        # Filter out non-numeric columns
        numeric_columns = data[columns].select_dtypes(include=['number']).columns

        if len(numeric_columns) < 2:
            # PCA requires at least two numeric columns
            return None

        # Apply PCA for dimensionality reduction
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numeric_columns])

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # Create a scatter plot for PCA components
        pca_plot_fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], title='PCA Plot', template=self.template_cl)
        pca_plot_html = plot(pca_plot_fig, output_type='div', include_plotlyjs=False)

        return pca_plot_html
    
    def select_top_columns22(self, columns, data):
        # Select the top N most important columns for visualizations
        top_columns = []

        for keyword in self.target_column_keywords:
            matching_columns = [column for column in columns if keyword in column.lower()]
            top_columns.extend(matching_columns)

        # Check if all columns are of object datatype
        if all(data[column].dtype == 'object' for column in columns):
            # If all columns are of object datatype, prioritize columns with the highest frequency
            mode_columns = data[columns].mode().iloc[0]
            if not mode_columns.empty:
                top_columns.extend(mode_columns.index)

        else:
            # Add additional columns if needed (you can modify this logic based on your requirements)
            # Example: Add columns with the highest correlation with the target variable
            correlation_matrix = data.select_dtypes(include=['number']).corr()
            if not top_columns:
                top_columns.extend(correlation_matrix.abs().nlargest(5, self.y_column).index)

        # Add remaining columns if necessary
        remaining_columns = set(columns) - set(top_columns)
        remaining_columns_list = list(remaining_columns)
        top_columns.extend(remaining_columns_list[:5 - len(top_columns)])  # Adjust the number of top columns as needed

        return top_columns[:5]