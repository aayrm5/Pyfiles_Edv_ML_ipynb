


from bokeh.charts import Scatter,output_file,show
import pandas as pd
import numpy as np



data_file=r'~/Dropbox/March onwards/Python Data Science/Data/cars.csv'



d=pd.read_csv(data_file)



d.head()



p=Scatter(d,x="Horsepower",y='CityMPG',
         title='Mileage Vs Horsepwoer',
         xlabel='Horsepower',
         ylabel="Mileage")



output_file('scatter.html')
show(p)



from bokeh.plotting import figure,output_file,show



p=figure(plot_width=500,plot_height=400)



p.diamond(d['CityMPG'],d['HighwayMPG'],color='blue',alpha=0.5,size=5)

output_file('scatter_plotting.html')
show(p)




from bokeh.models import HoverTool, ColumnDataSource



from bokeh.palettes import Viridis6



from bokeh.plotting import figure, show, output_notebook
from bokeh.sampledata.us_counties import data as counties
from bokeh.sampledata.unemployment import data as unemployment



counties = {
    code: county for code, county in counties.items() if county["state"] == "tx"
}

county_xs = [county["lons"] for county in counties.values()]
county_ys = [county["lats"] for county in counties.values()]






county_names = [county['name'] for county in counties.values()]
county_rates = [unemployment[county_id] for county_id in counties]
county_colors = [Viridis6[int(rate/3)] for rate in county_rates]

source = ColumnDataSource(data=dict(
    x=county_xs,
    y=county_ys,
    color=county_colors,
    name=county_names,
    rate=county_rates,
))



output_notebook()



TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"

p = figure(title="Texas Unemployment 2009", tools=TOOLS,
           x_axis_location=None, y_axis_location=None)
p.grid.grid_line_color = None

p.patches('x', 'y', source=source,
          fill_color='color', fill_alpha=0.7,
          line_color="white", line_width=0.5)

hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [
    ("Name", "@name"),
    ("Unemployment rate)", "@rate%"),
    ("(Long, Lat)", "($x, $y)"),
]



show(p)



import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource



N = 300
x = np.linspace(0, 4*np.pi, N)
y1 = np.sin(x)
y2 = np.cos(x)



source = ColumnDataSource(data=dict(x=x, y1=y1, y2=y2))



TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

s1 = figure(tools=TOOLS, plot_width=350, plot_height=350)
s1.scatter('x', 'y1', source=source)

s2 = figure(tools=TOOLS, plot_width=350, plot_height=350)
s2.circle('x', 'y2', source=source)



p = gridplot([[s1,s2]])
show(p)


