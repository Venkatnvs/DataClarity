from django.shortcuts import render,redirect
from django.http import JsonResponse
from concurrent.futures import ThreadPoolExecutor
from django.core.cache import cache
from django.views import View
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dashboard.views import BillionairesSearch
from dashboard.models import BillionairesWishlist
from django.contrib import messages

class CompareBillionaires(BillionairesSearch,View):
    def get(self,request):
        name1 = request.GET.get('name1', '')
        name2 = request.GET.get('name2', '')
        data = cache.get('billionaires_data')
        if data is None:
            with ThreadPoolExecutor() as executor:
                data = executor.submit(self.fetch_billionaires).result()
            cache.set('billionaires_data', data, timeout=3600)
        person1_data = next((b for b in data if name1.lower() in b['person']['name'].lower()), None)
        person2_data = next((b for b in data if name2.lower() in b['person']['name'].lower()), None)

        if person1_data is None or person2_data is None:
            return JsonResponse({'error': 'One or both persons not found'}, status=404)

        context = self.get_analysis_data(name1,name2,person1_data,person2_data)
        return render(request, 'main/compare_billionaires.html',context)
    
    def post(self,request):
        name1 = request.POST.get('name1', '')
        name2 = request.POST.get('name2', '')
        a = BillionairesWishlist.objects.create(name1=name1,name2=name2,user=request.user)
        a.save()
        messages.info(request,f'{name1} vs {name2} added to your wishlist')
        return redirect('main-bil-wishlist')
    
    def get_analysis_data(self,name1,name2,person1_data,person2_data):
        context = {}
        context['name1'] = name1
        context['name2'] = name2

        context['networth_graph'] = self.GetNetworthGraph(name1,name2,person1_data,person2_data)
        context['scatter_rank_graph'] = self.GetRankGraph(name1,name2,person1_data,person2_data)
        context['age_distribution_html'] = self.GetAgeGraph(name1,name2,person1_data,person2_data)
        context['industries_pie_chart1'], context['industries_pie_chart2'] = self.GetIndChatTwo(name1,name2,person1_data,person2_data)
        context['financial_assets_chart'] = self.GetFinancialAssetsBarChart(name1, name2, person1_data, person2_data)
        context['Source_chart'] = self.GetSourceBarChart(name1, name2, person1_data, person2_data)
        return context

    def GetNetworthGraph(self,name1,name2,person1_data,person2_data):
        labels = ['NetWorth']
        values_person1 = [person1_data['finalWorth']]
        values_person2 = [person2_data['finalWorth']]

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=labels, y=values_person1, name=name1))
        bar_fig.add_trace(go.Bar(x=labels, y=values_person2, name=name2))
        bar_fig.update_layout(title_text=f'Comparison between {name1} and {name2}', barmode='group')
        bar_html = bar_fig.to_html(full_html=False)
        return bar_html
    
    def GetRankGraph(self,name1,name2,person1_data,person2_data):
        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(x=[name1, name2], y=[person1_data['rank'], person2_data['rank']],
                                        mode='markers', name='Rank', text=["Rank"]))
        scatter_fig.update_layout(title_text=f'Scatter Plot - Rank Comparison', showlegend=True)
        scatter_html = scatter_fig.to_html(full_html=False)
        return scatter_html
    
    def GetAgeGraph(self,name1,name2,person1_data,person2_data):
        age_distribution_fig = go.Figure()
        age_distribution_fig.add_trace(go.Bar(x=[name1, name2], y=[self.calculate_age(person1_data['birthDate']),self.calculate_age(person2_data['birthDate'])]))
        age_distribution_fig.update_layout(title_text=f'Age Distribution', showlegend=True)
        age_distribution_html = age_distribution_fig.to_html(full_html=False)
        return age_distribution_html
    
    def GetIndustriesPieChart(self, name, person_data):
        industries_data = person_data.get('industries', [])
        industry_counts = {}
        for industry in industries_data:
            if industry in industry_counts:
                industry_counts[industry] += 1
            else:
                industry_counts[industry] = 1
        labels = list(industry_counts.keys())
        values = list(industry_counts.values())
        pie_chart_fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        pie_chart_fig.update_layout(title_text=f'Industries Distribution - {name}', showlegend=True)
        pie_chart_html = pie_chart_fig.to_html(full_html=False)
        return pie_chart_html

    def GetIndChatTwo(self, name1, name2, person1_data, person2_data):
        industries_pie_chart_person1 = self.GetIndustriesPieChart(name1, person1_data)
        industries_pie_chart_person2 = self.GetIndustriesPieChart(name2, person2_data)
        return industries_pie_chart_person1, industries_pie_chart_person2
    
    def GetFinancialAssetsBarChart(self, name1, name2, person1_data, person2_data):
        financial_assets_person1 = person1_data.get('financialAssets', [])
        financial_assets_person2 = person2_data.get('financialAssets', [])
        tickers_person1 = [asset['ticker'] for asset in financial_assets_person1]
        numberOfShares_person1 = [asset['numberOfShares'] for asset in financial_assets_person1]
        tickers_person2 = [asset['ticker'] for asset in financial_assets_person2]
        numberOfShares_person2 = [asset['numberOfShares'] for asset in financial_assets_person2]
        bar_chart_fig = go.Figure()
        bar_chart_fig.add_trace(go.Bar(x=tickers_person1, y=numberOfShares_person1, name=name1))
        bar_chart_fig.add_trace(go.Bar(x=tickers_person2, y=numberOfShares_person2, name=name2))
        bar_chart_fig.update_layout(title_text=f'Financial Assets Distribution - {name1} vs {name2}', xaxis_title='Ticker', yaxis_title='Number of Shares', barmode='group')
        bar_chart_html = bar_chart_fig.to_html(full_html=False)
        return bar_chart_html

    def GetSourceBarChart(self, name1, name2, person1_data, person2_data):
        industries_person1 = person1_data.get('source', '')
        industries_person2 = person2_data.get('source', '')
        industries_counts_person1 = {}
        industries_counts_person2 = {}
        for industry in industries_person1.split(', '):
            if industry in industries_counts_person1:
                industries_counts_person1[industry] += 1
            else:
                industries_counts_person1[industry] = 1
        for industry in industries_person2.split(', '):
            if industry in industries_counts_person2:
                industries_counts_person2[industry] += 1
            else:
                industries_counts_person2[industry] = 1
        labels_person1 = list(industries_counts_person1.keys())
        values_person1 = list(industries_counts_person1.values())
        labels_person2 = list(industries_counts_person2.keys())
        values_person2 = list(industries_counts_person2.values())
        bar_chart_fig = go.Figure()
        bar_chart_fig.add_trace(go.Bar(x=labels_person1, y=values_person1, name=name1))
        bar_chart_fig.add_trace(go.Bar(x=labels_person2, y=values_person2, name=name2))
        bar_chart_fig.update_layout(title_text=f'Industries Distribution - {name1} vs {name2}', xaxis_title='Industries', yaxis_title='Count', barmode='group')
        bar_chart_html = bar_chart_fig.to_html(full_html=False)
        return bar_chart_html