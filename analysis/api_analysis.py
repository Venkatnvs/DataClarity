from django.views import View
from django.shortcuts import render
import requests
from plotly.offline import plot
from plotly.graph_objs import Scatter, Bar, Pie, Histogram, Box
from django.contrib.auth.mixins import LoginRequiredMixin

class  AnalyzeApi(LoginRequiredMixin,View):
    def get(self,request):
        return render(request, 'main/analyze_api.html')
    
    def post(self,request):
        api_url = request.POST.get('api_url')
        api_key = request.POST.get('api_key', '')
        
        response = requests.get(api_url, headers={'Authorization': f'Bearer {api_key}'})
        api_data = response.json()

        plots = []
        for i in range(max(8, len(api_data[0]))):
            column_data = [entry[i] for entry in api_data]
            timestamps = [entry['timestamp'] for entry in api_data]

            if isinstance(column_data[0], (int, float)):
                plots.append(plot([Scatter(x=timestamps, y=column_data)], output_type='div'))

            elif isinstance(column_data[0], str):
                # Bar chart for categorical columns
                plots.append(plot([Bar(x=timestamps, y=column_data)], output_type='div'))

            elif isinstance(column_data[0], bool):
                # Pie chart for boolean columns
                labels = ['True', 'False']
                values = [column_data.count(True), column_data.count(False)]
                plots.append(plot([Pie(labels=labels, values=values)], output_type='div'))

            elif isinstance(column_data[0], (int, float)) and len(set(column_data)) > 1:
                # Histogram for numeric columns with more than one unique value
                plots.append(plot([Histogram(x=column_data)], output_type='div'))

            elif isinstance(column_data[0], (int, float)) and len(set(column_data)) == 1:
                # Box plot for numeric columns with only one unique value
                plots.append(plot([Box(y=column_data)], output_type='div'))

        return render(request, 'api_analysis/analysis_result.html', {'plots': plots})
