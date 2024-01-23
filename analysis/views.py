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
        try:
            context = self.analyze_data(data, uploaded_file)
        except Exception as e:
            context = {
                'error_message' :f"Unable to analyze the data\n {str(e)}",
            }
        return render(request, self.template_name, context)

    def analyze_data(self, data, uploaded_file):
        context = {}

        data_summary = data.describe().transpose().to_html(classes='table table-hover table-sm table-responsive nvs-table-data-ds')
        context['data_summary'] = data_summary

        useful_columns = self.get_useful_columns(data)
        y_column = self.auto_identify_target_column(data, useful_columns)
        self.y_column = y_column
        top_columns = self.select_top_columns(useful_columns, data, y_column)
        if not top_columns or not y_column:
            context['error_message'] = "Unable to analyze the data"
            return context

        plots_html = self.create_plots(data, top_columns, y_column)
        context['plots_html'] = plots_html

        report = f"Number of Rows: {len(data)}\n\nData Head:\n{data.head()}"
        context['report'] = report

        forecast_data = self.forecast_data(data, y_column)
        context['forecast_data'] = forecast_data

        trend_data, trend_line = self.linear_regression(data, top_columns[0], y_column)
        context['trend_data'] = trend_data
        context['trend_line'] = trend_line

        anomalies_s, anomalies = self.detect_anomalies(data, top_columns[0], y_column)
        context['anomalies_s'] = anomalies_s
        context['anomalies'] = anomalies

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
                continue 

            if pd.api.types.is_numeric_dtype(data[column]):
                top_columns.append(column)
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                top_columns.append(column)
            elif pd.api.types.is_object_dtype(data[column]):
                top_columns.append(column)

        return top_columns[:8]

    def auto_identify_target_column(self, data, columns):

        for column in data.columns:
            if any(keyword in column.lower() for keyword in self.target_column_keywords):
                return column

        numeric_columns = data.select_dtypes(include=['number']).columns
        if not numeric_columns.empty:
            return numeric_columns[-1]

        for column in columns:
            if column in data.select_dtypes(include=['number']).columns:
                return data[columns].mean().idxmax()
            elif column in data.select_dtypes(include=['datetime']).columns:
                return data[columns].min().idxmin()
            elif column in data.select_dtypes(include=['object']).columns:
                return data[column].mode()[0]

        return None
    
    def create_plots(self, data, columns, y_column):
        plots_html = []

        if y_column not in data.columns:
            return plots_html

        for column in columns:
            num_unique_values = data[column].nunique()

            if num_unique_values == 1:
                fig = px.pie(data, names=column, title=f'Pie Chart: {column} Distribution', template=self.template_cl)
                self.add_unique_items_annotation(fig, data[column].nunique())
                plots_html.append(fig.to_html(full_html=False))
            elif num_unique_values == 2:
                fig = px.bar(data, x=column, y=y_column, color=column, title=f'Bar Plot: {column} vs {y_column} by {column}', template=self.template_cl)
                self.add_unique_items_annotation(fig, data[column].nunique())
                plots_html.append(fig.to_html(full_html=False))
            elif pd.api.types.is_numeric_dtype(data[column]):
                fig = px.box(data, x=column, y=y_column, points='all', title=f'Box Plot: {column} vs {y_column}', template=self.template_cl)
                self.add_unique_items_annotation(fig, data[column].nunique())
                plots_html.append(fig.to_html(full_html=False))
            elif pd.api.types.is_object_dtype(data[column]):
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
            forecast_model = ExponentialSmoothing(data[y_column], seasonal='add', seasonal_periods=12)
            forecast_result = forecast_model.fit()
            forecast_data = forecast_result.forecast(12)  # Forecasting the next 12 periods
            return forecast_data.tolist()  # Convert to list for easy integration into template
        except Exception as e:
            return None

    def linear_regression(self, data, x_column, y_column):
        try:
            model = LinearRegression()
            X = data[[x_column]]
            y = data[y_column]
            model.fit(X, y)
            trend_line = pd.DataFrame({x_column: data[x_column], 'Trend': model.predict(X)})
            scatter_fig = px.scatter(data, x=x_column, y=y_column, title=f'Scatter Plot: {x_column} vs {y_column}', template=self.template_cl)
            trend_fig = px.line(trend_line, x=x_column, y='Trend', title=f'Linear Regression Trend: {x_column} vs {y_column}', template=self.template_cl)

            scatter_html = plot(scatter_fig, output_type='div', include_plotlyjs=False)
            trend_html = plot(trend_fig, output_type='div', include_plotlyjs=False)

            return scatter_html, trend_html
        
        except Exception as e:
            return None, None

    def detect_anomalies(self, data, x_column, y_column):
        try:
            clf = IsolationForest(contamination=0.05)
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
        histograms = self.generate_histograms(data, selected_columns)

        correlation_matrix = self.generate_correlation_matrix(data, selected_columns)

        box_plots = self.generate_box_plots(data, selected_columns)

        heatmap = self.generate_heatmap(data, selected_columns)

        pair_plots = self.generate_pair_plots(data, selected_columns)

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
                hist_fig = px.histogram(data, x=column, title=f'Histogram: {column}', color=column, template=self.template_cl)
                histograms_html.append(hist_fig.to_html(full_html=False))

        return histograms_html

    def generate_correlation_matrix(self, data, columns):
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
                box_fig = px.box(data, y=column, title=f'Box Plot: {column}', color=column, template=self.template_cl)
                box_plots_html.append(box_fig.to_html(full_html=False))

        return box_plots_html

    def generate_heatmap(self, data, columns):
        heatmap_fig = px.imshow(data[columns], aspect='auto', color_continuous_scale='viridis', title='Heatmap', template=self.template_cl)
        heatmap_html = heatmap_fig.to_html(full_html=False)

        return heatmap_html

    def generate_pair_plots(self, data, columns):
        try:
            pair_plot_fig = px.scatter_matrix(data[columns], dimensions=columns, template=self.template_cl)
            pair_plot_html = pair_plot_fig.to_html(full_html=False)

            return pair_plot_html
        except Exception as e:
            return None

    def generate_pca_plot(self, data, columns):
        numeric_columns = data[columns].select_dtypes(include=['number']).columns

        if len(numeric_columns) < 2:
            return None

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numeric_columns])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        pca_plot_fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], title='PCA Plot', template=self.template_cl)
        pca_plot_html = plot(pca_plot_fig, output_type='div', include_plotlyjs=False)

        return pca_plot_html
    
    def select_top_columns22(self, columns, data):
        top_columns = []

        for keyword in self.target_column_keywords:
            matching_columns = [column for column in columns if keyword in column.lower()]
            top_columns.extend(matching_columns)

        if all(data[column].dtype == 'object' for column in columns):
            mode_columns = data[columns].mode().iloc[0]
            if not mode_columns.empty:
                top_columns.extend(mode_columns.index)

        else:
            correlation_matrix = data.select_dtypes(include=['number']).corr()
            if not top_columns:
                top_columns.extend(correlation_matrix.abs().nlargest(5, self.y_column).index)

        remaining_columns = set(columns) - set(top_columns)
        remaining_columns_list = list(remaining_columns)
        top_columns.extend(remaining_columns_list[:5 - len(top_columns)])

        return top_columns[:5]