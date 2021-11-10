import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.base import is_classifier, is_regressor
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["figure.figsize"] = (15,15)
import math
import seaborn as sns
import shap
from datetime import datetime
import time
from ipywidgets.embed import embed_minimal_html
import umap
from pandas_profiling import ProfileReport
from sklearn.neighbors import kneighbors_graph
from prophet import Prophet
from umap import UMAP
from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.preprocessing import Binarizer,FunctionTransformer, KBinsDiscretizer, KernelCenterer, LabelBinarizer, LabelEncoder, MinMaxScaler,MaxAbsScaler,\
                                  QuantileTransformer, Normalizer, OneHotEncoder, OrdinalEncoder,PowerTransformer, RobustScaler, SplineTransformer,StandardScaler, PolynomialFeatures
from sklearn.decomposition import DictionaryLearning,FastICA, IncrementalPCA, KernelPCA, MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF,PCA,SparsePCA, FactorAnalysis,\
                                  TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding, TSNE
from sklearn.pipeline import make_pipeline
from sklearn.utils import estimator_html_repr
#import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import  BaseEnsemble,RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding, ExtraTreesClassifier, ExtraTreesRegressor,\
                          BaggingClassifier, BaggingRegressor, IsolationForest, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier,\
                          AdaBoostRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, HistGradientBoostingClassifier,\
                          HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV, Hinge, Huber, HuberRegressor, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC,\
                             LinearRegression, Log, LogisticRegression, LogisticRegressionCV, ModifiedHuber,MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso,\
                             MultiTaskLassoCV, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, \
                             QuantileRegressor, Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV, SGDClassifier, SGDRegressor, SGDOneClassSVM, SquaredLoss,TheilSenRegressor, \
                             RANSACRegressor, PoissonRegressor,GammaRegressor,TweedieRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB,ComplementNB, CategoricalNB
from scipy.stats import ttest_ind, ttest_1samp
from  plotly.offline  import plot_mpl
import plotly.tools as ptools
import networkx as nx
from prophet.plot import plot_plotly, plot_components_plotly
import calendar
from prophet.utilities import regressor_coefficients
import plotly.express as px
import base64
import numpy as np
import pandas as pd
from io import StringIO
import io
#from keplergl import KeplerGl
import hdbscan
import datetime
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances

import dash_cytoscape as cyto
import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output# Load Data
from dash.dash_table.Format import Format, Scheme, Trim

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from joblib import Memory
from shutil import rmtree
from sklearn import svm, datasets
from sklearn.metrics import auc,confusion_matrix,classification_report
from sklearn.metrics import RocCurveDisplay,ConfusionMatrixDisplay
#from sklearn.metrics import plot_roc_curve,plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from skopt import BayesSearchCV, gp_minimize, forest_minimize, gbrt_minimize
from skopt.searchcv import BayesSearchCV as BSCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from sklearn.feature_selection import RFECV
set_config(display='diagram')
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib
import ipywidgets as widget
from sklearn.base import clone
import plotly.figure_factory as ff

class pipemaker2:
    def __init__(self, df,ipt_pipe, target ,*, height = 'auto', width = 'auto'):
        self.pipe_list = []
        self.df = df
        self.TG = target
        self.check = 0
        self.cached_pipe = 0
        self.location = 0
        self.memory = 0
        self.optimized_pipe = (0, 0)
        self.input_pipe = ipt_pipe

    def Pipe(self):
        return clone(self.input_pipe)

    def Cache_pipe(self):
        self.location = 'cachedir'
        self.memory = Memory(location=self.location, verbose=0)
        self.cached_pipe = self.Pipe().set_params(memory = self.memory)

    def release_cache(self):
        self.memory.clear(warn=True)
        rmtree(self.location)
        del self.memory

    def export_kwards(self):
        return self.Pipe().get_params()
    def fit_transform(self):
        return self.ColumnTransform().fit_transform(self.df)
    def fit_predict(self):
        return self.Pipe().fit_predict(self.df, self.df[self.TG])
    def fit(self):
        return self.Pipe().fit(self.df, self.df[self.TG])

    def RFECV(self):
        preprocessed_df = pd.DataFrame(self.Pipe()['preprocessing'].fit_transform(self.df))

        if self.optimized_pipe[1] == 0:
            selector = RFECV(self.Pipe()['classifier'], step=1, cv=KFold(10, shuffle= True)).fit(preprocessed_df, self.df[self.TG])
        else:
            selector = RFECV(self.optimized_pipe[0]['classifier'], step=1, cv=KFold(10, shuffle= True)).fit(preprocessed_df, self.df[self.TG])

        hX = np.array( range(1, len(selector.grid_scores_) + 1))
        hY= selector.grid_scores_
        H = pd.DataFrame(np.array([hX, hY]).T, columns = ['Number of parameters', 'Cross Validation Score'])

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(hX, hY)
        plt.show()
        return pd.DataFrame([selector.ranking_, selector.support_], columns = preprocessed_df.columns, index = ['Ranking', 'support'])

    def make_skpot_var(self, param, temperature = 3, distribution = 'uniform', just_classifier = False): #'log-uniform'
        value = self.export_kwards()[param]
        if just_classifier == True: name = param.split('__')[1]
        else: name = param

        if value == 0 or value ==1: return

        if type(value) == int:
            if value == -1: return Integer(1, 200, name = name)
            lower_bondary = int(value/temperature)
            if lower_bondary < 2: lower_bondary = 2
            upper_bondary = int(value*temperature) + lower_bondary
            #if value <= 1: return Real(1e-3, 1, distribution ,name = name)
            return Integer(lower_bondary, upper_bondary, distribution ,name = name)

        if type(value) == float:
            if value == -1: return Real(1, 200, name = name)
            if value <= 1: return Real(1e-3, 1, distribution ,name = name)
            lower_bondary = value/temperature
            if lower_bondary < 2: lower_bondary = 2
            upper_bondary = value*temperature + lower_bondary
            return Real(lower_bondary, upper_bondary, distribution ,name = name)

    def skopt_classifier_space(self, just_classifier = False):
        dic = self.export_kwards()
        classifier_params = [x for x in  dic.keys()
                             if x.find('classifier__') != -1
                             and  x.find('silent') == -1
                             and  x.find('n_jobs') == -1
                             and x.find('bagging_fraction') == -1
                             and x != 'classifier__subsample'
                             and x != 'classifier__validation_fraction'] # and
        SPACE = [self.make_skpot_var(i, just_classifier = just_classifier) for i in classifier_params]
        SPACE = [x for x in SPACE if x if x != None ]
        return SPACE

    def objective(self, params):
        classifier = self.Pipe().set_params(**{dim.name: val for dim, val in zip(self.skopt_classifier_space(), params)})
        return -np.mean(cross_val_score(classifier, self.df, self.df[self.TG], cv = StratifiedKFold(n_splits = 5, shuffle=True)))

    def objective_just_classifier(self, params, metric , cv_method ):
        return -np.mean(cross_val_score(self.cached_pipe['classifier'].set_params(**{dim.name: val for dim, val in zip(self.skopt_classifier_space(just_classifier = 1), params)}),
                                        self.transformed_opt,
                                        self.target_opt,
                                        scoring = metric,
                                        cv = cv_method,
                                        n_jobs = -1))

    def objective_cached(self, params):
        return -np.mean(cross_val_score(self.cached_pipe.set_params(**{dim.name: val for dim, val in zip(self.skopt_classifier_space(), params)}),
                                        self.df,
                                        self.df[self.TG],
                                        cv = StratifiedKFold(n_splits = 5, shuffle=True)))


    def optimize_classifier(self, n_calls = 50, cache = False):
        if cache:
            self.Cache_pipe()
            result = gp_minimize(self.objective_cached, self.skopt_classifier_space() , n_calls=n_calls)
            self.release_cache()
        else: result = gp_minimize(self.objective, self.skopt_classifier_space() , n_calls=n_calls)
        #plot_convergence(result)
        #_ = plot_objective(result, n_points=n_calls)
        #print(result.fun)
        return {'result': result, 'best_params': self.get_params(result, self.skopt_classifier_space() )}

    def fast_optimize_classifier(self, n_calls = 50,  is_classifier = True):
        self.Cache_pipe()

        self.transformed_opt = self.cached_pipe['preprocessing'].fit_transform(self.df)
        self.target_opt = self.df[self.TG]

        if is_classifier:
            cv_method = StratifiedKFold(n_splits = 5, shuffle=True)
            metric    = 'f1_weighted'
        else:
            cv_method = KFold(n_splits = 5, shuffle=True)
            metric    = 'r2'

        result = gp_minimize(lambda x: self.objective_just_classifier(x, metric, cv_method), self.skopt_classifier_space(just_classifier = True) , n_calls=n_calls)
        self.release_cache()

        best_params = self.get_params(result, self.skopt_classifier_space(just_classifier = True))
        best_params = {'classifier__'+ i[0]:i[1] for i in best_params.items()}

        self.optimized_pipe = (self.Pipe().set_params(**best_params), 1)

        return {'result': result, 'best_params':best_params}

    def get_params(self, result_object, space):
        try:
            return { i.name: result_object.x[num] for  num, i in enumerate(space) }
        except:
            raise

    def Vis_Cluster(self, method):
        transformed = self.Pipe()['preprocessing'].fit_transform(self.df)
        classsification = method.fit_predict(transformed)  #(*args, **kwds)
        end_time = time.time()
        palette = sns.color_palette('deep', np.unique(classsification).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in classsification]
        plt.scatter(transformed.T[0], transformed.T[1], c=colors, s = MinMaxScaler(feature_range=(30, 300)).fit_transform(self.df[self.TG].values.reshape(-1, 1)) , **{'alpha' : 0.5,  'linewidths':0})
        frame = plt.gca()
        for num, spine in enumerate(frame.spines.values()):
            if num == 1 or num == 3: spine.set_visible(False)
        plt.title('Clusters found by {}'.format(str(method)), fontsize=24)
        plt.show()
        return

    def Evaluate_model(self):
        tprs = []
        aucs = []
        prd = []
        tru = []
        mean_fpr = np.linspace(0, 1, 100)
        X = self.df.copy()
        y = self.df[self.TG]
        if self.optimized_pipe[1] == 0: clf = self.Pipe()
        else: clf = self.optimized_pipe[0]
        fig, ax = plt.subplots(1, 2, figsize = (20,10))
        try:
            for i, (train, test) in enumerate(StratifiedKFold(n_splits=5, shuffle=True).split(X, y)):
                clf.fit(X.iloc[train], y.iloc[train])
                viz = RocCurveDisplay.from_estimator(clf, X.iloc[test], y.iloc[test],
                                     name='ROC fold {}'.format(i),
                                     alpha=0.3, lw=1, ax=ax[0])
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            ax[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                    label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax[0].plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax[0].fill_between(mean_fpr, tprs_lower, tprs_upper, color='steelblue', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')

            ax[0].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
            #       title="Receiver operating characteristic example")
            ax[0].legend(loc="lower right")
        except: pass#print('non-binary classifier')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        try:
            ConfusionMatrixDisplay.from_estimator(clf.fit(X_train, y_train), X_test, y_test,
                                         display_labels=['negative detection', 'positive detection'],
                                         cmap=plt.cm.Blues, ax = ax[1])
            ax[1].grid(False)
        except: pass # print('is it a regressor?')
        fig.tight_layout()
        try:
            report = classification_report(clf.predict(X_test), y_test, output_dict=True) # target_names=['Negative detection', 'Positive detection']
        except: #### report for regression
            if self.optimized_pipe[1] == 0: clf = self.Pipe()
            else: clf = self.optimized_pipe[0]
            report = cross_validate(clf, X, y, cv=5,  scoring=('neg_mean_absolute_percentage_error','r2','explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error'))
            fig, ax = plt.subplots(1, 1, figsize = (10,10))
            fig.tight_layout()
        return report, fig

    def named_preprocessor(self):
        naming_features = []
        for transformer in self.Pipe()['preprocessing'].transformers:
            transformed = ColumnTransformer(transformers = [transformer]).fit_transform(self.df)
            if transformed.shape[1] == len(transformer[2]):
                naming_features += list(transformer[2])
            else:
                naming_features += [transformer[0] +'__'+ str(i) for i in range(transformed.shape[1]) ]
        if self.optimized_pipe[1] == 0: clf = self.Pipe()
        else: clf = self.optimized_pipe[0]
        return pd.DataFrame(clf['preprocessing'].fit_transform(self.df), columns = naming_features)

    def Shapley_feature_importance(self):
        if self.optimized_pipe[1] == 0: clf = self.Pipe()
        else: clf = self.optimized_pipe[0]
        shap.initjs()
        dat_trans = self.named_preprocessor()
        explainer = shap.TreeExplainer(clf['classifier'].fit(dat_trans, self.df[self.TG])) #,feature_perturbation = "tree_path_dependent"
        shap_values = explainer.shap_values(dat_trans)

        #### force-plot
        a = [_force_plot_html(explainer.expected_value[i], shap_values[i], dat_trans) for i in len(shap_values)]

        ### dependence matrix
        ivalues = explainer.shap_interaction_values(dat_trans)
        figdm, axdm = plt.subplots(len( dat_trans.columns),  len(dat_trans.columns), figsize=(15, 15))
        d = {i: name for i,name in enumerate(dat_trans.columns)}
        for i in d.keys():
            for j in d.keys():
                shap.dependence_plot((d[i], d[j]), ivalues[1], dat_trans, ax = axdm[i,j], show = False)
        return (a,  figdm) #fig,


#JupyterDash.infer_jupyter_proxy_config()

cyto.load_extra_layouts()
height, width = [500,500]
canvas_width = 500
canvas_height = round(height * canvas_width / width)
scale = canvas_width / width

def plotly_cyt(d):
    edges = [{'data': {'weight': i['data']['weight'], 'source': str(i['data']['source']), 'target': str(i['data']['target'])}}  for i in d['edges']]
    nodes = [{'data': {k:i['data'][k] for k in ('id', 'value', 'name') }, 'position' : dict(zip(('x', 'y'),i['data']['data']))} for i in d['nodes']]
    return nodes + edges

def plotly_cyt2(G):
    d = nx.cytoscape_data(G)['elements']
    pos = nx.spring_layout(G)
    edges = [{'data': {'weight': i['data']['weight'], 'source': str(i['data']['source']), 'target': str(i['data']['target'])}}  for i in d['edges']]
    nodes = [{'data': {k:i['data'][k] for k in ('id', 'value', 'name') }, 'position' : dict(zip(('x', 'y'),j))} for i,j in zip(d['nodes'], list(pos.values()))]
    return nodes + edges

def plotly_cyt3(G):
    d = nx.cytoscape_data(G)['elements']
    pos = nx.spring_layout(G)
    edges = [{'data': {'weight': i['data']['weight'], 'source': str(i['data']['source']), 'target': str(i['data']['target'])}}  for i in d['edges']]
    nodes = [{'data': {**{k:i['data'][k] for k in ('id', 'value', 'name') }, **{'degree': degree[1]}} , 'position' : dict(zip(('x', 'y'),j))}
             for i,j,degree in zip(d['nodes'], list(pos.values()), list(G.degree))]
    return nodes + edges

def make_colormap_clustering(column, palette, continuous, data):
    if not continuous:
        lut = dict(zip(sorted(data[column].unique()), sns.color_palette(palette, len(data[column].unique()))))
    else: lut = sns.color_palette(palette, as_cmap=True)
    return data[column].map(lut)


def _force_plot_html(*args):
    force_plot = shap.force_plot(*args, matplotlib=False, figsize=(18, 18))
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return html.Iframe(srcDoc=shap_html, height='1800', width='1800',style={"border": 0})#

def mplfig2html(figure):
    pic_IObytes2 = io.BytesIO()
    figure.savefig(pic_IObytes2,  format='png')
    figure.clear()
    pic_IObytes2.seek(0)
    return  html.Img(src ='data:image/png;base64,{}'.format(base64.b64encode(pic_IObytes2.read()).decode()))

def mpl2plotlyGraph(figure):
    return dcc.Graph(ptools.mpl_to_plotly(figure)) #image_height: int=600,image_width: int=800

# Build App
#app = JupyterDash(__name__, external_stylesheets=[dbc.themes.MINTY]) #FLATLY, LUMEN, SUPERHERO
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

def convert2cytoscapeJSON(G):
    # load all nodes into nodes array
    final = {}
    final["nodes"] = []
    final["edges"] = []
    for node in G.nodes():
        nx = {}
        nx["data"] = {}
        nx["data"]["id"] = node
        nx["data"]["label"] = node
        final["nodes"].append(nx.copy())
    #load all edges to edges array
    for edge in G.edges():
        nx = {}
        nx["data"]={}
        nx["data"]["id"]=edge[0]+edge[1]
        nx["data"]["source"]=edge[0]
        nx["data"]["target"]=edge[1]
        final["edges"].append(nx)
    return json.dumps(final)


upload_tab = [
    dbc.Row(dbc.Col(dbc.Container([
        html.H3("Send complete dataset directly", className="display-4"),
        dcc.Upload(id='upload_dataset_directly',children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={'width': '100%', 'height': '80px',  'lineHeight': '80px', 'font-size': '20px',   'borderWidth': '1px', 'borderStyle': 'dashed',  'borderRadius': '5px',   'textAlign': 'center', 'margin': '10px'},multiple=False),
        html.Div(id='direct_dataframe_upload_name')
    ],className="h-100 p-5 bg-light border rounded-4 g-0"), width = 12), justify="center",className="h-100 p-5 bg-light border rounded-4 g-0 gap-2"),

    dbc.Row(dbc.Col(dbc.Container([
        html.H4("Send qPCR files and metadata", className="display-4"),
        html.H4("qPCR files", className="display-5"),
        html.P('We are expecting csv files from an export file from a cfx96',className="lead",),
        html.Hr(className="my-2"),
        dbc.Row([
        dbc.Col(html.H5("Column of qPCR files to merge with habitat metadata:") , width = 4),
        dbc.Col(dcc.Dropdown(options = [{"label": "Sample", "value": 'Sample'}] , value = 'Sample', id='qpcrdf', disabled = True), width = 3)]),
        dcc.Upload(id='upload-qPCR2',children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={'width': '100%',
                'height': '80px',
                'lineHeight': '80px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'font-size': '20px',
                'borderRadius': '5px',
                'justify-content': 'center',
                'textAlign': 'center',
                'margin': '10px'}, multiple=True),
        html.Div(id='qpcr-data-upload'),
        html.H4("Habitat metadata", className="display-5"),
        html.P('You probably have a separate file with Lat, Lon and other environmental parameters',className="lead",),
        html.Hr(className="my-2"),
        dbc.Row([
        dbc.Col(html.H5("Column of Habitat metadata file to merge with qPCRs:") , width = 4),
        dbc.Col(dcc.Dropdown(id='habitatdf'), width = 3)]),
        dcc.Upload(id='upload-habitat',children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={'width': '100%',
                'height': '80px',
                'lineHeight': '80px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'font-size': '20px',
                'justify-content': 'center',
                'textAlign': 'center',
                'margin': '10px'},multiple=True),
        html.Div(id='habitat-data-upload')
    ],className="h-100 p-5 bg-light border-2 rounded-2 g-0"),width = 12),justify="center",className="h-100 p-5 bg-light border rounded-4 g-0 gap-2"),
    #dbc.Row(dbc.Col(dbc.Container([],className="h-100 p-5 bg-light rounded-2 g-0", fluid = True), width = 11),justify="center",className="g-0")


]

merge_tab = [
    dbc.Container([
        html.H2("Dataset overview ", className="display-4"),
        #html.P('Look for parameters that have unexpected behavior, dataset size and other possible concerns with data integrity',className="lead",),
        html.Hr(className="my-2"),html.P(""),
        dcc.Loading(id="loading-1",type="default", children=html.Div(id='Merged_df', style = {'justify-content': 'center', 'margin': '0 auto', 'width': '95%'} ) )
    ],className="h-100 p-5 bg-light border rounded-3 g-0", fluid = True),
]



VIS=[ dbc.Row([
           dbc.Col(
               [dbc.Row([
                   dbc.Container([
                   html.H5("Choose dataset:"),
                   dcc.Dropdown(options=[{'label':i, 'value':i} for i in ['df','df with umap','selected points of umap plot']],
                                         value='df', multi=False, id = 'dataset_mapping',clearable=False),
                   html.H5("latitude column:"),
                   dcc.Dropdown(options=[],value=[], multi=False, id = 'lat_mapping',clearable=False),
                   html.H5("longitude column:"),
                   dcc.Dropdown(options=[],value=[], multi=False, id = 'lon_mapping',clearable=False),
                   html.H5("time:"),
                   dcc.Dropdown(options=[{'label':'none', 'value':'none'}],value='none',  multi=False, id = 'time_mapping', clearable=False),
                   html.H5("colormap:"),
                   dcc.Dropdown(options=[{'label':'none', 'value':'none'}],value='none',  multi=False, id = 'color_mapping', clearable=False),
                   html.H5("size:"),
                   dcc.Dropdown(options=[{'label':'none', 'value':'none'}],value='none',  multi=False, id = 'size_mapping', clearable=False),
                   html.Hr(style= {'margin-bottom': '30px'}),
                   dbc.Button("Run map", color="info", size = 'lg', className="d-grid gap-2", id='run_keplermap') ],className="h-100 p-5 bg-light border rounded-3 g-0 d-grid", fluid = True),

               ])],width=2)  ,
           dbc.Col([dcc.Loading(id="loading-keplermap",type="default", children= html.Div( id = 'keplermap', style = {'overflow': 'hidden'})) ], width=10, style = {'overflow': 'clip'})],  className="g-0")] #




kep_tab=[ dbc.Row([
           dbc.Col(
               [dbc.Row([
                   dbc.Container([
                       html.H5("what are the continous columns for the UMAP?", id = 'kep_tab_continuous_columns_target'),
                       dbc.Popover([ dbc.PopoverHeader("how we look at continuous data"),dbc.PopoverBody("https://umap-learn.readthedocs.io/en/latest/basic_usage.html")],target="kep_tab_continuous_columns_target",trigger="hover",),
                       dcc.Dropdown(options=[],value=[], multi=True, id = 'UMAP_cont'),
                       html.H5("what are the categorical columns for the UMAP?", id = 'kep_tab_cat_columns_target'),
                       dbc.Popover([ dbc.PopoverHeader("how we look at categorical data"),dbc.PopoverBody("see https://umap-learn.readthedocs.io/en/latest/composing_models.html#diamonds-dataset-example")],target="kep_tab_cat_columns_target",trigger="hover",),
                       dcc.Dropdown(options=[],value=[], multi=True, id = 'UMAP_cat'),
                       html.H5("Do you want to fit the UMAP to a feature?", id = 'keep_tab_metric_learn'), #https://umap-learn.readthedocs.io/en/latest/supervised.html
                       dbc.Popover([ dbc.PopoverHeader("fitting umap to feature"),dbc.PopoverBody("https://umap-learn.readthedocs.io/en/latest/supervised.html")],target="keep_tab_metric_learn",trigger="hover",),
                       dcc.Dropdown(options=[],value=[], multi=False, id = 'UMAP_y'),
                       html.H5("How many neighboors for the UMAP to use?", id = 'keep_tab_nneighboors'),
                       dbc.Popover([ dbc.PopoverHeader("n neighboors parameter"),dbc.PopoverBody("This parameter controls how UMAP balances local versus global structure in the data. It does this by \
                       constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data. \
                       This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture),\
                       while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, \
                       losing fine detail structure for the sake of getting the broader of the data. _ see https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors")],target="keep_tab_nneighboors",trigger="hover",),
                       dbc.Input(id="n_neighboors", type="number", value = 15, min = 10, max = 1000), #https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
                       html.H5('Type of scaling to use:', id= 'kep_tab_scale'),
                       dbc.Popover([ dbc.PopoverHeader("Should I scale my data?"),dbc.PopoverBody("The default answer is yes, but, of course, the real answer is “it depends”. \
                       If your features have meaningful relationships with one another (say, latitude and longitude values) then normalising per feature is not a good idea. \
                       For features that are essentially independent it does make sense to get all the features on (relatively) the same scale. \
                       The best way to do this is to use pre-processing tools from scikit-learn. All the advice given there applies as sensible preprocessing for UMAP,\
                       and since UMAP is scikit-learn compatible you can put all of this together into a scikit-learn pipeline.")],target="kep_tab_scale",trigger="hover",),
                       dbc.RadioItems(id="UMAP_radio",
                        options=[
                            {"label": "No Standardization", "value": 1},
                            {"label": "Standard scaler", "value": 2},
                            {"label": "Pipeline from machine learning tab","value": 3}],value = 2,
                            labelCheckedStyle={"color": "#223c4f", 'font-size': '14px'},
                            labelStyle = {}, style = {'font-size': '14px', 'margin' : '8px', 'margin-left': '20px' ,'transform':'scale(1.1)'}, switch=True,
                            inputStyle = { }
                                     ),
                      dbc.Button("Generate UMAP", color="info", size = 'lg', className="d-grid gap-2", id='UMAP_start') ],className="h-100 p-5 bg-light border rounded-3 g-0 d-grid", fluid = True),
                      dbc.Popover([ dbc.PopoverHeader("what is UMAP?"),dbc.PopoverBody("see https://umap-learn.readthedocs.io/en/latest/how_umap_works.html \nhttps://umap-learn.readthedocs.io/en/latest/scientific_papers.html\nhttps://umap-learn.readthedocs.io/en/latest/faq.html#what-is-the-difference-between-pca-umap-vaes")],target="UMAP_start",trigger="hover",),


               ])],width=2)  ,
           dbc.Col([dcc.Loading(id="loading-umap",type="default", children= dcc.Tabs([
               dcc.Tab(label = 'umap-view', children = [html.Div(dcc.Graph(id='UMAP_view'), style = {'height': '1000px', 'width' : '1500px','margin-left':'30px'}),html.Div( id = 'umap_selected_stats', style = {'width': '98%'})] ),
               dcc.Tab(label = 'heatmap/cytoscape', children = html.Div( id = 'cytoscape', style = {'justify-content': 'center'} )),
               dcc.Tab(label = 'hdbscan clustering', children = html.Div(id='graph') ),

           ], style = {'justify-content': 'center','display': 'flex' ,'width': '100%','margin-left': '12px','overflow': 'clip'})) ], width=10, style = {'overflow': 'clip'})],  className="g-0")] #

#className="nav nav-pills"      , className="g-0"         autosize=False

time_series_tab = [
    dbc.Row([
        dbc.Col( dbc.Container([
            html.H5("Target column"),
            dcc.Dropdown(options=[],value=[], multi=False, id = 'prophet_y'),
            html.H5("Datetime column"),
            dcc.Dropdown(options=[],value=[], multi=False, id = 'prophet_ds'),
            html.Hr(style= {'margin-bottom': '3px'}),
            html.H5("Additional regressors"),
            dcc.Dropdown(options=[],value=[], multi=True, id = 'prophet_regressors'),
            html.Hr(style= {'margin-bottom': '3px'}),
            html.H5('Rolling average'),
            html.H6('number of days'),
            dbc.Input(id="prophet_rolling_average", type="number", value = 0, min = 0, max = 366, step = 0.25),
            html.Hr(style= {'margin-bottom': '3px'}),
            html.H5("Growth"),
            dcc.Dropdown(options=[
                {"label": "logistic", "value": 'logistic'},
                {"label": "flat", "value": 'flat'},
                {"label": "linear", "value": 'linear'}
            ],value='linear', multi=False,id = 'prophet_growth'),
            html.H5("Target maximum value"),
            dbc.Input(id="prophet_cap", type="number", value = 1, step = .01),
            html.H5("Target minimum value"),
            dbc.Input(id="prophet_floor", type="number", value = 0, step = .01),
            html.Hr(style= {'margin-bottom': '3px'}),
            html.H5('Seasonnality'),
            html.H6('frequency'),
            dbc.Checklist( options = [
                {"label": "Yearly", "value": 'yearly_seasonality'},
                {"label": "Weekly", "value": 'weekly_seasonality'},
                {"label": "Daily", "value": 'daily_seasonality'},
            ]  ,value=['yearly_seasonality'], id = 'prophet_seasonality' ,
                          style = {'font-size': '14px', 'margin' : '2px', 'margin-left': '20px' ,'transform':'scale(1.)'}, switch=True),
            html.H6('mode'),
            dcc.Dropdown(options=[
                {"label": "additive", "value": 'additive'},
                {"label": "multiplicative", "value": 'multiplicative'}
            ], multi=False,id = 'seasonality_mode', value = 'additive'),
            html.H6('scale'),
            dbc.Input(id="season_prior", type="number", value = 10, min = 1, max = 100),
            html.Hr(style= {'margin-bottom': '3px'}),
            html.H5('Change points'),
            html.H6('quantity'),
            dbc.Input(id="prophet_n_change_points", type="number", value = 25, min = 0, max = 100,step =1),
            html.H6('scale'),
            dbc.Input(id="changepoint_prior", type="number", value = .05, min = 0, max = 10., step = 0.01),
            html.H6('range'),
            dbc.Input(id="changepoint_range", type="number", value = .8, min = 0.1, max = 1., step = 0.01),




        ],className="h-100 p-5 bg-light border rounded-3 g-0", fluid = True), width = 2),
        dbc.Col(dcc.Loading(id="loading-prophet",type="default", children=html.Div(id='prophet_plots', style = {'justify-content': 'center', 'margin': '0 auto', 'width': '100%'} ), style= {'margin-top': '100px'})),
        dbc.Col( dbc.Container([
            html.H5('Forecast'),
            html.H6('prediction range'),
            dcc.DatePickerRange(id= 'prophet_future_dates', display_format='MMM DD YYYY'),
            html.Hr(style= {'margin-bottom': '50px'}),
            html.H6('remove these month'),
            dcc.Dropdown(options=[ {"label":  calendar.month_name[num], "value": num} for num in range(1,12)],value=[], multi=True,id = 'prophet_remove_months'),
            html.H6('remove these days of the week'),
            dcc.Dropdown(options=[ {"label":  day_name, "value": num} for num,day_name in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])],
                         value=[], multi=True,id = 'prophet_remove_days_of_the_week'),
            html.H6('remove these hours of the day'),
            dcc.Dropdown(options=[ {"label":  str(num)+':00-'+str(num+1)+':00', "value": num} for num in range(0,24)],value=[], multi=True,id = 'prophet_remove_hours'),
            html.Hr(style= {'margin-bottom': '50px'}),
            dbc.Button("Run forecast", color="info", size = 'lg', id='run_prophet')
        ],className="h-100 p-5 bg-light border rounded-3 g-0", fluid = True), className="g-0", width = 2)


    ], className="g-0", style={'margin-bottom': '10px'})


]

sklearn_preprocessor_list = ['Binarizer','FunctionTransformer', 'KBinsDiscretizer', 'KernelCenterer', 'LabelBinarizer', 'LabelEncoder', 'MinMaxScaler',
                             'MaxAbsScaler','QuantileTransformer', 'Normalizer', 'OneHotEncoder', 'OrdinalEncoder', 'PowerTransformer', 'RobustScaler', 'SplineTransformer',
                              'StandardScaler', 'PolynomialFeatures']
sklearn_decomposition_list = ['DictionaryLearning','FastICA', 'IncrementalPCA', 'KernelPCA', 'MiniBatchDictionaryLearning', 'MiniBatchSparsePCA',
                              'NMF','PCA','SparsePCA', 'FactorAnalysis','TruncatedSVD', 'LatentDirichletAllocation']
sklearn_manifold_list = ['LocallyLinearEmbedding', 'Isomap', 'MDS', 'SpectralEmbedding', 'TSNE']
transformers = sklearn_preprocessor_list + sklearn_decomposition_list + sklearn_manifold_list + ['UMAP', 'passthrough']
transformer_options = [ {'label': x, 'value': x } for x in  transformers]

sklearn_ensemble_list  = ['BaseEnsemble','RandomForestClassifier', 'RandomForestRegressor', 'RandomTreesEmbedding', 'ExtraTreesClassifier', 'ExtraTreesRegressor',
                          'BaggingClassifier', 'BaggingRegressor', 'IsolationForest', 'GradientBoostingClassifier', 'GradientBoostingRegressor', 'AdaBoostClassifier',
                          'AdaBoostRegressor', 'VotingClassifier', 'VotingRegressor', 'StackingClassifier', 'StackingRegressor', 'HistGradientBoostingClassifier',
                          'HistGradientBoostingRegressor']
sklearn_linear_model_list = ['ARDRegression', 'BayesianRidge', 'ElasticNet', 'ElasticNetCV', 'Hinge', 'Huber', 'HuberRegressor', 'Lars', 'LarsCV', 'Lasso', 'LassoCV',
                             'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LinearRegression', 'Log', 'LogisticRegression', 'LogisticRegressionCV', 'ModifiedHuber',
                             'MultiTaskElasticNet', 'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuitCV',
                             'PassiveAggressiveClassifier', 'PassiveAggressiveRegressor', 'Perceptron', 'QuantileRegressor', 'Ridge', 'RidgeCV', 'RidgeClassifier',
                             'RidgeClassifierCV', 'SGDClassifier', 'SGDRegressor', 'SGDOneClassSVM', 'SquaredLoss', 'TheilSenRegressor',
                            'RANSACRegressor', 'PoissonRegressor', 'GammaRegressor', 'TweedieRegressor']
sklearn_naive_bayes_list = ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB', 'CategoricalNB']
classifier_list = ['LGBMClassifier', 'LGBMRegressor'] + sklearn_ensemble_list + sklearn_naive_bayes_list + sklearn_linear_model_list
classifier_options = [ {'label': x, 'value': x } for x in  classifier_list]


ML_tab = [
   dbc.Row([
       dbc.Col(
           [dbc.Container([
                dbc.Row([
                   dbc.Col([ html.H5("number of transformers:")]),
                   dbc.Col([#dcc.Dropdown(options=[ {'label': str(x), 'value': str(x)} for x in range(10)],value='2', multi=False,clearable=False, id = 'n_tabs')
                            dbc.Input(id="n_tabs", type="number", value = 1, min = 1, max = 10)
                           ]),
                   dbc.Col([html.H5("Target:")]),
                   dbc.Col([dcc.Dropdown(options=[],value=[], multi=False, id = 'ML_target',clearable=False)]),
                   dbc.Col([html.H5("Classifier:", id = 'ml_tab_classifier'), dbc.Popover([ dbc.PopoverHeader("chosing a classifier"),dbc.PopoverBody('see: \
                   https://scikit-learn.org/stable/supervised_learning.html#supervised-learning\n https://lightgbm.readthedocs.io/en/latest/Quick-Start.html ')],target="ml_tab_classifier",trigger="hover",)]),
                   dbc.Col([dcc.Dropdown(options=classifier_options ,value = 'LGBMClassifier',  multi=False, id = 'clf_disp', clearable=False)]) ]), #)],className="h-100 p-5 bg-light border rounded-3 g-0", fluid = True),
            dbc.Row([dbc.Col(
                   [html.H5("Columns to be transformed:")] +
                   [ dcc.Dropdown(options= ['0'], value = ['0'],multi=True,clearable=False, id = 'Columns_'+ str(i))  for i in range(3)], id = 'preprocessing_columns'),
            dbc.Col(
                   [html.H5("Column transformers:", id = 'ml_tab_column_trans')] + #https://scikit-learn.org/stable/modules/preprocessing.html#
                   [ dcc.Dropdown(options= transformer_options, value = ['passthrough'], multi=True,clearable=False, id = 'ColumnTransformer_'+ str(i))  for i in range(3)], id = 'preprocessing_functions'),
            dbc.Popover([ dbc.PopoverHeader("preprocessing the data"),dbc.PopoverBody("see:\n https://scikit-learn.org/stable/modules/preprocessing.html\n\
                   https://scikit-learn.org/stable/modules/decomposition.html#decompositions#\nhttps://scikit-learn.org/stable/modules/clustering.html#clustering")],target="ml_tab_column_trans",trigger="hover",)
                                   ])],className="h-100 p-5 bg-light border rounded-1 g-0",fluid = True)


           ],width=6, id='ml_user_input') ] + [dbc.Col([dbc.Button("Update Pipeline", color="info", size = 'lg', className="d-grid gap-2", id='submit_pipe'),
                                                         html.Div(id = 'show_pipeline', style ={'width': '50%','borderWidth': '0px' ,'border': 'white'})],
                                                        width = 6)], className="g-0",justify="center", style = {'font-size': '12px', 'margin' : '5px' }),
    dbc.Row([dbc.Col(
        dbc.Container([
           dbc.Row([ html.H2("Testing the pipeline", style ={'margin': '20px'})]), #,justify="center"
            dbc.Row([dbc.Col([html.H5("Number of runs for hyperparameter optimization (use < 10 for no optimization):", id = 'ml_tab_tunning')], width = 3),
                      dbc.Popover([ dbc.PopoverHeader("Tunning the model"),dbc.PopoverBody("here we use scikit optimize's bayesian optimization to tune the hyperparameters\
                      https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html")],target="ml_tab_tunning",trigger="hover",),
                    dbc.Col([dbc.Input(id="slider_hyperopt", type="number", value = 5, min = 10, max = 1000)], width = 1)], className="g-0", style={'margin-bottom': '10px'}), #
            dbc.Row([dbc.Button("Run pipeline", color="info", size = 'lg', className="d-grid gap-2",  id='run_ML')], className = 'd-grid g-0'),
            dcc.Loading(id="loading-ml",type="default", children=html.Div(id = 'ml_results', style = {'justify-content': 'center', 'margin': '0 auto', 'width': '1760', 'height' : '220px'}),
                                 style= {'margin-top': '-300px','justify-content': 'center'})],className="h-100 p-5 bg-light border rounded-3 g-0", fluid = True)
                     , width = 12,  style = {'justify-content': 'center', 'height' : '220px'}) ], className="g-0")
]


# html.Iframe(srcDoc = ret_map._repr_html_().decode(), height='1280', width='2350') iframe for html representation of pipeline sklearn
tab_style = {
    "background": "#223c4f",
    'color': "#6cc3d5",
    'text-transform': 'lowercase',
    'border': '#223c4f',
    'font-size': '12px',
    'font-weight': 100, #200
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '0px',
    'height': '20px',
    'display': 'flex',

    #'padding':'6px'
}

tab_selected_style = {
    "background": "#153751",
    'color': 'white',
    'text-transform': 'uppercase',
    'font-size': '12px',
    'font-weight': 100, #200
    'align-items': 'center',
    'height': '20px',
    'justify-content': 'center',
    'display': 'flex',
    #'box-shadow': '60px 0 #223c4f, -60px 0 solid #223c4f',
    'border-style': 'solid #223c4f',
    'border-color': '#223c4f',
    'border-width': '0',
    #'border-radius': '50px'
}

subtab_style = {'height': '20px','display': 'flex','font-size': '12px'}

app.layout = html.Div([
    dbc.NavbarSimple([], brand = 'eDNA dashbord - Genomic Variation Laboraty UCDavis', brand_style ={'color': "white",'font-size': '14px'} ,
                     style = { 'align-items': 'left','justify-content': 'left', 'font-size': '14px', 'height': '32px'},
                    color = "#223c4f"),
    dcc.Store(id='all_qPCR_concat', storage_type='memory'), #storage_type='local'
    dcc.Store(id='habitatcsv', storage_type='memory'),  #df_with_umap
    dcc.Store(id='df', storage_type='memory'),
    dcc.Store(id='df_with_umap', storage_type='memory'),
    dcc.Store(id='umap_select_columns', storage_type='memory'),
    dcc.Store(id='selected_points_umap', storage_type='memory'), #html.Table(id='all_dfs')    selected_points_umap
    dcc.Tabs([
        dcc.Tab(label = 'Dataset', children = upload_tab , style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label = 'Quality Control', children = merge_tab , style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Exploratory Data Analysis', children=kep_tab, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Geoposition', children=VIS, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Time Series', children=time_series_tab, style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Machine Learning', children=ML_tab, style=tab_style, selected_style=tab_selected_style)]) , #,className="nav nav-pills"
        ])

def get_cq_list(x, df):
    a = df[df.Sample ==  x].Cq.values
    return [60 if np.isnan(x) else x for x in a]

def get_det_list(x, df):
    a = df[df.Sample ==  x].Call.values
    return [1 if x=='(+) Positive' else 0 for x in a]

def FIND_Better(row, column, df):
    series = df[df.index == str(row['SiteID'])][column]
    if series.shape[0] == 0: return -1
    return series.iloc[0]
cyto.load_extra_layouts()

@app.callback(Output(component_id= 'Merged_df', component_property ='children'),
              Output(component_id= 'df', component_property ='data'),
              Output(component_id= 'direct_dataframe_upload_name', component_property = 'children'),
              Input('habitatdf', 'value'),
              Input('upload_dataset_directly', 'contents'),
              State('upload_dataset_directly', 'filename'),
              State('upload_dataset_directly', 'last_modified'),
              State('all_qPCR_concat', 'data'),
              State('habitatcsv', 'data'))
def merge_csv_update_spreadsheet(hab, up_content, up_filename,  up_date , df_qpcr_json, df_hab_json): #qpcr,
    ctx = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if  hab != None and ctx == 'habitatdf': # and qpcr != None:
        try : left_merge, right_merge =  hab, 'Sample' #qpcr
        except:
            return   html.Div(), html.Hr(className="my-2"), html.Div(),

        try: df, df_hab = pd.read_json(df_qpcr_json), pd.read_json(df_hab_json)
        except Exception as e:
            return   html.H5('no data'), html.Hr(className="my-2"), html.Div(),

        describe = df.groupby('Sample').describe(include = 'all', percentiles = [])
        frequencies = pd.DataFrame(describe['Cq']['count']/describe['Fluor']['count'], columns = ['eDNA_frq']) #### this is not working properly for some reason?!
        frequencies = frequencies.fillna(-1)
        frequencies['min_Cq'] = describe['Cq']['min']
        frequencies['N_samples'] = describe['Fluor']['count']
        frequencies = frequencies.reset_index()
        frequencies['list_of_CQs'] = frequencies.Sample.apply(lambda x: get_cq_list(x, df))
        frequencies['CQ_average'] =  frequencies.list_of_CQs.apply(lambda x: np.array(x).mean())
        frequencies['list_of_detections'] = frequencies.Sample.apply(lambda x: get_det_list(x, df))
        frequencies['eDNA_detection_average'] =  frequencies.list_of_detections.apply(lambda x: np.array(x).mean())
        frequencies['eDNA_binary'] = frequencies['eDNA_detection_average'].apply(lambda x: 1 if x > .1 else 0)
        frequencies = frequencies.fillna(60)
        final = df_hab.merge(frequencies,left_on = left_merge, right_on = right_merge, how = 'inner' ) #
        return  dbc.Container([ html.H2("Overview of your dataset", className="display-3"),
                               html.Iframe(srcDoc = ProfileReport(final,  correlations=None,interactions=None).to_html(), height='900', width='1600')],className="h-100 p-5 bg-light border rounded-3 g-0", fluid = True),  final.to_json(), html.Div() #interactions=None,
    elif up_content != None and ctx == 'upload_dataset_directly':
        content_type, content_string = up_content.split(',')
        decoded = base64.b64decode(content_string)
        if '.csv' in up_filename:
            final = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif '.tsv' in up_filename:
            final = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t')
        else: return html.Div('file type not accepted, is it a csv or a tsv?'),html.Div(), html.Div([  html.H6(up_filename), html.Hr() ])  #minimal=True
        return  [ html.Iframe(srcDoc = ProfileReport(final,correlations=None,interactions=None).to_html(), height='900', width='1600')],final.to_json(), html.Div([html.H6(up_filename), html.Hr()]) #2350 interactions=None,

    else: return html.Div(),html.Div(), html.Div()



classifier_list = ['LGBMClassifier', 'LGBMRegressor'] + sklearn_ensemble_list + sklearn_naive_bayes_list + sklearn_linear_model_list

def inpt_children_to_pipe(columns, funcs, classif):
    C = [x['props']['value'] for x in columns[1:]]
    F = [x['props']['value'] for x in funcs[1:]]

    if classif == 'LGBMClassifier' or  classif == 'LGBMRegressor':
        #classifier_function = getattr(lightgbm, classif)(boosting_type='gbdt',  subsample=1.0) #boosting_type='gbdt', bagging_fraction = 0
        classifier_function = globals()[classif](boosting_type='gbdt',  subsample=1.0)
    else: classifier_function = globals()[classif]()
    return Pipeline(steps = [('preprocessing', make_pipe(C, F)), ('classifier', classifier_function)])

def make_pipe(columns_list, transformer_list):
    simplfy = []
    for num, (cols, trans) in enumerate(zip(columns_list, transformer_list) ):
        sub_smp = []
        for x in trans:
            if x[0].isupper() == True:
                if x in sklearn_decomposition_list: sub_smp += [globals()[x](n_components = 2)]
                else: sub_smp += [globals()[x]()]
            else: sub_smp += [x]
        simplfy += [tuple([str(num), make_pipeline(*sub_smp), tuple(cols)])]
    return ColumnTransformer(simplfy)
    #simplfy =[ tuple([str(num), make_pipeline(*[locals()[x]() if x[0].isupper() == True else x for x in trans ]), tuple(cols)]) for num, (cols, trans) in enumerate(zip(columns_list, transformer_list) )]
    #return ColumnTransformer(simplfy)


@app.callback(Output(component_id= 'ml_results', component_property ='children'),
              Input(component_id = 'run_ML', component_property = 'n_clicks'),
              State(component_id= 'preprocessing_functions', component_property ='children'),
              State(component_id= 'preprocessing_columns', component_property ='children'),
              State(component_id = 'clf_disp', component_property = 'value'),
              State(component_id = 'df', component_property = 'data'),
              State(component_id = 'ML_target', component_property = 'value'),
              State(component_id = 'slider_hyperopt', component_property = 'value'))
def run_ML(clicked, f_list, c_list, val, data, target, ncalls):
    pipe = inpt_children_to_pipe(c_list,f_list, val)
    if 'CV' in val: ncalls = 2
    try: df = pd.read_json(data,convert_dates = False)
    except: return html.Div()
    Maj = pipemaker2(df, pipe, target)
    try:
        opt_results = Maj.fast_optimize_classifier(n_calls= int(ncalls))
        new_pipe2 = [html.Iframe(srcDoc = estimator_html_repr(Maj.optimized_pipe[0]),height='360', width='920', hidden = 'hidden')]
    except:
        try:
            opt_results = Maj.fast_optimize_classifier(n_calls= int(ncalls), is_classifier= False)
            new_pipe2 = [html.Iframe(srcDoc = estimator_html_repr(Maj.optimized_pipe[0]), height='360', width='920', hidden = 'hidden')]
        except:
            new_pipe2 = [html.Iframe(srcDoc = estimator_html_repr(Maj.Pipe()), height='360', width='920', hidden = 'hidden')]
            Maj = pipemaker2(pd.read_json(data,convert_dates = False), inpt_children_to_pipe(c_list,f_list, val), target)
    try:
        scores, fig  = Maj.Evaluate_model()
        rev_table = pd.DataFrame(scores).T.reset_index().round(3)
        graph_part = mplfig2html(fig)
        #scoreshtml = [dash_table.DataTable( data=rev_table.to_dict('records'), columns=[{'name': str(i), 'id': str(i)} for i in rev_table.columns]),graph_part] #, style_table={'overflowX': 'auto' #style_cell={'minWidth': '180px', 'width': '180px', 'maxWidth': '180px','overflow': 'hidden','textOverflow': 'ellipsis'}
        scoreshtml = [dcc.Graph(figure=ff.create_table(rev_table)),graph_part]

    except: scoreshtml =  [html.H4('Failed evaluate scores: is it a regressor?', className="display-3") ,html.H4('Failed evaluate scores: is it a regressor?', className="display-3") ]

    ##### shapley graphs
    if Maj.optimized_pipe[1] == 0: clf = Maj.Pipe()
    else: clf = Maj.optimized_pipe[0]

    new_pipe = html.Iframe(srcDoc = estimator_html_repr(clf), height='360', width='920', hidden = True)
    #fig, ax = plt.subplots(figsize=(15, 15))
    shap.initjs()
    dat_trans = Maj.named_preprocessor()
    try:
        explainer = shap.TreeExplainer(clf['classifier'].fit(dat_trans, Maj.df[Maj.TG]), dat_trans) ######## added dat_trans here ____________________remove if breaks!!!
        shap_values = explainer.shap_values(dat_trans, check_additivity=False)        #,feature_perturbation = "tree_path_dependent"
    except:
        explainer = shap.Explainer(clf['classifier'].fit(dat_trans, Maj.df[Maj.TG]), dat_trans)
        shap_values = explainer.shap_values(dat_trans)

    #### summary plot
    fig_summary, ax = plt.subplots(figsize=(15, 15))
    shap.summary_plot(shap_values,dat_trans, plot_type='bar',plot_size=(10,10), max_display=20,show= False)
    sumhtml = [mplfig2html(fig_summary)]

    #### force-plot
    try: a = [_force_plot_html(explainer.expected_value[i], shap_values[i], dat_trans) for i in range(len(shap_values))]
    except: a = [_force_plot_html(explainer.expected_value, shap_values, dat_trans) ]
    #a = []
    ### dependence matrix
    try:
        ivalues = shap.TreeExplainer(clf['classifier'].fit(dat_trans, Maj.df[Maj.TG])).shap_interaction_values(dat_trans)
        figdm, axdm = plt.subplots(len( dat_trans.columns),  len(dat_trans.columns), figsize=(15, 15))
        shap.summary_plot(ivalues, dat_trans, show= False)
        ####erase here if necessary
        figdm = plt.gcf()
        figdm.set_figheight(15)
        figdm.set_figwidth(15)
        figdm.tight_layout()
        fig2html = mplfig2html(figdm)
    except:
        fig2html = html.H6("Shapley interaction matrix only available for tree-based models")

    #### heatmap
    try:
        try : shap.plots.heatmap(explainer(dat_trans), show= False)
        except : shap.plots.heatmap(explainer(dat_trans), show= False, check_additivity=False)
        fig1 = plt.gcf()
        fig1.set_figheight(15)
        fig1.set_figwidth(15)
        fig1.tight_layout()
        fig1html = mplfig2html(fig1)
        heatmapfigs = [fig1html]
    except:
        heatmapfigs = [html.H6('heatmap is only available in binary classification')]

    if val == "LGBMClassifier" or val == 'LGBMRegressor':
        decision_tree, ax = plt.subplots(1,1, figsize=(15, 15))
        lgbmfig = []

    else:
        lgbmfig = []

    figure_names =  (['scores','roc-auc & cm'] if is_classifier(globals()[val]()) else ['score'] )+ ['feature importance'] + ['force-plot feat'+ str(i) for i in range(len(a))] + ['heatmap', 'feature interaction'] + ['decision_tree' for x in lgbmfig]
    ml_all_figures = (scoreshtml if is_classifier(globals()[val]()) else scoreshtml[:1] ) + sumhtml +a +heatmapfigs + [fig2html] + lgbmfig
    ml_result_tabs = dcc.Tabs([dcc.Tab(children = html.Div(content, style = {'justify-content': 'center', 'margin': '0 auto', 'width': '1760px', 'height' : '1100px'}), label = name)
                               for name,content in zip(figure_names, ml_all_figures)],
                              style = {'justify-content': 'center', 'margin': '0 auto', 'width': '100%'})

    return [ml_result_tabs]+ new_pipe2
    #return dbc.Container(scoreshtml+ sumhtml +a +heatmapfigs + [fig2html]+ new_pipe2)#+ new_pipe2 #fig1html,fig2html

@app.callback(Output(component_id= 'show_pipeline', component_property ='children'),
              Input(component_id= 'preprocessing_functions', component_property ='children'),
              Input(component_id= 'preprocessing_columns', component_property ='children'),
              Input(component_id = 'clf_disp', component_property = 'value'),
              Input(component_id= 'ml_results', component_property ='children'),
              Input(component_id = 'submit_pipe', component_property = 'n_clicks') )
def html_pipe(f_list, c_list, val, ml_children, clicked):
    ctx = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if ctx != 'ml_results':
        pipe = inpt_children_to_pipe(c_list,f_list, val)
        return html.Iframe(srcDoc = estimator_html_repr(pipe),  height='360', width='920',style = {'border-style':'none', 'frameborder':'none'})
    else:
        try: ret = ml_children[-1]['props']['srcDoc']
        except: return html.Iframe(srcDoc = estimator_html_repr(inpt_children_to_pipe(c_list,f_list, val)), height='360', width='920',  style = {'border-style':'none', 'frameborder':'none'})
    return html.Iframe(srcDoc = ret,  height='360', width='920', style = {'border-style':'none', 'frameborder':'none'}) #1150



@app.callback(Output(component_id= 'preprocessing_functions', component_property ='children'),
              Output(component_id= 'preprocessing_columns', component_property ='children'),
              Input('n_tabs', 'value'),
              Input(component_id = 'df', component_property = 'data'),
              State(component_id= 'preprocessing_functions', component_property ='children'),
              State(component_id= 'preprocessing_columns', component_property ='children') )
def reajust_number_of_column_transformers_ML(val,data,oldf, oldc ):
    if int(val) > len(oldf) - 1 :
        new_func = oldf + [ dcc.Dropdown(options= transformer_options,value = ['passthrough'] ,multi=True,clearable=True, id = 'ColumnTransformer_'+ str(i)) for i in range(len(oldf)-1, int(val))]
    elif int(val) < len(oldf) - 1:
        new_func =  oldf[:int(val)+1]
    else:
        new_func =  oldf

    try: df = pd.read_json(data,convert_dates = False)
    except: df = pd.DataFrame([0], columns = ['0'])
    col_cat =  [x for x in df.columns if str(df[x].dtype) == 'int64']
    col_num = [x for x in df.columns if str(df[x].dtype) == 'float64']
    sorted_vals = [{'label': x, 'value': x} for x in col_num + col_cat] + [ {'label': x, 'value': x} for x in  df.columns if x not in ['Unnamed: 0']+ col_num + col_cat ]
    new_c = [oldc[0]]+[ dcc.Dropdown(options= sorted_vals, value = '0' , multi=True,clearable=True, id = 'ColumnSelector_'+ str(i)) for i in range(int(val))]
    return new_func, new_c



@app.callback(Output(component_id= 'qpcr-data-upload', component_property ='children'),
              Output(component_id= 'qpcrdf', component_property ='options'),
              Output('all_qPCR_concat', 'data'),
              Input('upload-qPCR2', 'contents'),
              State('upload-qPCR2', 'filename'),
              State('upload-qPCR2', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = []
        allqpcrs = []
        for contents, filename, date in  zip(list_of_contents, list_of_names, list_of_dates):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), skiprows=19)
                    ex_header = pd.read_csv(io.StringIO(decoded.decode('utf-8')),  nrows=9, header=None).iloc[:, :2]
                    for i in range(9): df[ex_header.iloc[i,0]] = ex_header.iloc[i,1]
                    df['Bottle_replicate'] = df.Sample.apply(lambda x: x.split('.')[1] if '.' in x else -1)
                    df.Sample = df.Sample.apply(lambda x: x.split('.')[0])
                    allqpcrs += [df]
            except Exception as e:
                print(e)
                children += [html.Div(['There was an error processing this file. Is it a CSV?' ])]

            children += [html.Div([  html.H6(filename), html.Hr() ]) ]

        qpcr_files_concat =  pd.concat(allqpcrs).reset_index(drop = True)
        vals = [ {'label': x, 'value': x} for x in  qpcr_files_concat.columns]
        #merger_id = dcc.Dropdown( options=vals ,  searchable=False, id = 'qpcr_dropdown' , value = 'Sample')
        return children, vals, qpcr_files_concat.to_json()
    return html.Div(),[{"label": "Sample", "value": 'Sample'}], pd.DataFrame(np.zeros([1,1])).to_json()


@app.callback(Output(component_id= 'habitat-data-upload', component_property ='children'),
              Output(component_id= 'habitatdf', component_property ='options'),
              Output('habitatcsv', 'data'),
              Input('upload-habitat', 'contents'),
              State('upload-habitat', 'filename'),
              State('upload-habitat', 'last_modified'))
def update_output_hab(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = []
        for contents, filename, date in  zip(list_of_contents, list_of_names, list_of_dates):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:   allhabs = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                if 'tsv' in filename:   allhabs = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep= '\t')
                elif 'xls' in filename: allhabs = pd.read_excel(io.BytesIO(decoded))

            except Exception as e:
                print(e)
                children += [html.Div(['There was an error processing this file.' ])]

            children += [html.Div([  html.H6(filename), html.Hr()])   ]

        vals = [ {'label': x, 'value': x} for x in  allhabs.columns]
        #merger_id = dcc.Dropdown( options=vals )#, id = 'hab_dropdown') ,  searchable=False
        return children, vals, allhabs.to_json()
    return html.Div(),[], pd.DataFrame(np.zeros([1,1])).to_json()

@app.callback(Output(component_id= 'UMAP_cat', component_property ='options'),  Output(component_id= 'UMAP_cat', component_property ='value'),
              Output(component_id= 'UMAP_y', component_property ='options'),  Output(component_id= 'UMAP_y', component_property ='value'),
              Output(component_id= 'UMAP_cont', component_property ='options'), Output(component_id= 'UMAP_cont', component_property ='value'),
              Output(component_id= 'ML_target', component_property ='options'),  Output(component_id= 'ML_target', component_property ='value'),
              Output(component_id= 'prophet_y', component_property ='options'),  Output(component_id= 'prophet_y', component_property ='value'),
              Output(component_id= 'prophet_ds', component_property ='options'),  Output(component_id= 'prophet_ds', component_property ='value'),
              Output(component_id= 'prophet_regressors', component_property ='options'),  Output(component_id= 'prophet_regressors', component_property ='value'),
              Input(component_id= 'Merged_df', component_property ='children'), Input(component_id= 'df', component_property ='data'))
def update_UMAP_and_ML_select_columns(inpt, data): #, columns_list_id
    #if data != {'namespace': 'dash_html_components', 'props': {'children': None}, 'type': 'Div'} and data != None and inpt['type'] != 'Div':
    try:
        df = pd.read_json(data)
        vals = [ {'label': x, 'value': x} for x in  df.columns if x not in ['Unnamed: 0']]
        col_cat =  [x for x in df.columns if str(df[x].dtype) == 'int64']
        col_num = [x for x in df.columns if str(df[x].dtype) == 'float64']
        col_object = [x for x in df.columns if (str(df[x].dtype) in ['object', 'datetime64[ns]'] )]
        sorted_vals = [{'label': x, 'value': x} for x in col_num + col_cat] + [ {'label': x, 'value': x} for x in  df.columns if x not in ['Unnamed: 0']+ col_num + col_cat ]
        if len(col_object) > 0:
            if 'date' in col_object: col_object = 'date'
            elif 'datetime' in col_object: col_object = 'datetime'
            else: col_object = col_object[0]
        vals_object = [ {'label': x, 'value': x} for x in  df.columns  if (str(df[x].dtype) in ['object', 'datetime64[ns]'] )]
        vals_plus_umap = sorted_vals +  [{'label': 'UMAP_'+str(x), 'value': 'UMAP_'+str(x)} for x in range(1,3)]
        #prep_cols =  [columns_list_id[0]]+[ dcc.Dropdown(options= [{'label': x, 'value': x} for x in df.columns], value = df.columns[0] , multi=True,clearable=True, id = 'ColumnSelector_'+ str(i)) for i in range(len(columns_list_id)+1)]

        return sorted_vals, col_object,  sorted_vals, [], sorted_vals,col_num +col_cat,  sorted_vals, ['eDNA frq'], vals_plus_umap, [], vals_object,col_object,  vals_plus_umap, []

    except:
        return [], [], [], [], [],[], [], [], [], [],[], [],[], []  #, columns_list_id  str(fixed_dataset.date.dtype) == 'object'


@app.callback(Output(component_id= 'UMAP_view', component_property ='figure'),
              Output(component_id= 'df_with_umap', component_property ='data'),
              Output(component_id= 'graph', component_property ='children'),
              Output(component_id= 'cytoscape', component_property ='children'),
              Input('UMAP_start', 'n_clicks'),
              State('UMAP_cat', 'value'),
              State('UMAP_cont', 'value'),
              State('UMAP_y', 'value'),
              State('n_neighboors', 'value'),
              State('UMAP_radio', 'value'),
              State(component_id= 'preprocessing_columns', component_property ='children'),
              State(component_id= 'preprocessing_functions', component_property ='children'),
              State(component_id = 'clf_disp', component_property = 'value'),
              State(component_id= 'df', component_property ='data'))
def generate_UMAP(clicked, cat_labels, cont_labels,y ,n_nb, radio_val,MLcolumns, MLfuncs, MLclassif, dataframe_json):
    umap_list = []
    if dataframe_json != None:
        df = pd.read_json(dataframe_json).dropna(subset = cont_labels+cat_labels)
        if y == None or y == []:
            if len(cont_labels) > 0:
                if radio_val == 2: preprocessed_data = StandardScaler().fit_transform(df[cont_labels])
                if radio_val == 1: preprocessed_data = df[cont_labels]
                if radio_val == 3: preprocessed_data = inpt_children_to_pipe(MLcolumns, MLfuncs, MLclassif)['preprocessing'].fit_transform(df)
                umap_list += [umap.UMAP(n_neighbors = n_nb).fit(preprocessed_data)]
            if len(cat_labels) > 0:
                try: umap_list += [umap.UMAP(metric="jaccard", n_neighbors=150).fit(make_pipeline(OneHotEncoder()).fit_transform(df[cat_labels]))]
                except: umap_list += [umap.UMAP(metric="jaccard", n_neighbors=150).fit(make_pipeline(OrdinalEncoder(), MinMaxScaler()).fit_transform(df[cat_labels]))]
        else:# len(y) > 0:#:
            if len(cont_labels) > 0:
                if radio_val == 2: preprocessed_data = StandardScaler().fit_transform(df[cont_labels])
                if radio_val == 1: preprocessed_data = df[cont_labels]
                if radio_val == 3: preprocessed_data = inpt_children_to_pipe(MLcolumns, MLfuncs, MLclassif)['preprocessing'].fit_transform(df)
                umap_list +=[umap.UMAP(n_neighbors = n_nb).fit(preprocessed_data,y=df[y])]
            if len(cat_labels) > 0:
                try: umap_list += [umap.UMAP(metric="jaccard", n_neighbors=150).fit(make_pipeline(OneHotEncoder()).fit_transform(df[cat_labels]),y=df[y])]
                except: umap_list += [umap.UMAP(metric="jaccard", n_neighbors=150).fit(make_pipeline(OrdinalEncoder(), MinMaxScaler()).fit_transform(df[cat_labels]),y=df[y])]

        if len(umap_list) > 1: UMAP = umap_list[0] + umap_list[1]
        elif len(umap_list) == 1: UMAP = umap_list[0]
        else: return html.Div(), pd.DataFrame(np.zeros([1,1])).to_json() , html.Div()
        umap_df = pd.DataFrame(UMAP.embedding_, index = df.index, columns = ['UMAP_1', 'UMAP_2'])
        df = pd.concat([df, umap_df], axis = 1)
        cluster = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
        df['hdbscan'] =  cluster.fit_predict(df[['UMAP_1', 'UMAP_2']])
        df.columns = [x.replace(' ', '_') for x in df.columns]
        dfscatter = df.copy()
        dfscatter['hdbscan'] = dfscatter['hdbscan'].apply(str) #------- covert to str ------------
        dfscatter = dfscatter.reset_index()

        #------------------------------------------------- generate graph of distances! ----------------------
        default_stylesheet_cyto = [
        {'selector': '[degree < 15]','style': {'background-color': '#223c4f','label': 'data(id)','width': "30%",'height': "30%" }},
        {'selector': 'edge','style': {'line-color': '#223c4f', "mid-target-arrow-color": "red", "mid-target-arrow-shape": "vee" }},
        {'selector': '[degree >= 15]', 'style': {'background-color': 'red','label': 'data(id)', 'width': "40%", 'height': "40%"}}   ]
        if df.shape[0] < 200:
        #cyt = nx.cytoscape_data(cluster.minimum_spanning_tree_.to_networkx())['elements']
            cyt = nx.from_scipy_sparse_matrix(kneighbors_graph(umap_df, 2, mode = 'distance', include_self= False, n_jobs = -1), create_using=nx.DiGraph)

            cytodisplay2 = cyto.Cytoscape(id='cytoscape', layout={'name': 'cose'},style={'width': '1000px', 'height': '90%'},
                                          stylesheet = default_stylesheet_cyto,
                                          elements = plotly_cyt3(cyt)) #{'width': '2000px', 'height': '1000px'}
        else:
            df_colors_sns = pd.DataFrame(MinMaxScaler(feature_range = (-2,2)).fit_transform(dfscatter[['UMAP_1','UMAP_2']]), columns = ['UMAP_1','UMAP_2'])
            colors_sns = pd.concat([make_colormap_clustering('hdbscan', 'tab10',0, dfscatter),
                                    make_colormap_clustering('UMAP_1', 'PiYG',1, df_colors_sns).apply(lambda x: x[:-1]),
                                    make_colormap_clustering('UMAP_2', 'PiYG',1, df_colors_sns)], axis = 1)
            sns.clustermap(dfscatter[[x.replace(' ', '_') for x in cont_labels]], figsize=(15,14),cmap = sns.diverging_palette(20, 220, as_cmap=True), z_score = 1, cbar_pos = None, vmax = 2, vmin = -2,
                           row_colors =colors_sns , dendrogram_ratio=(.2, .1)) #col_cluster=False
            fig1 = plt.gcf()
            fig1.tight_layout()
            cytodisplay2 = mplfig2html(fig1) #mplfig2html(fig1) --------------- edited here---------------------------


        #### image from hdbscan
        pic_IObytes = io.BytesIO()
        fig = plt.figure(figsize = [16,6], dpi = 100)
        ax = fig.add_subplot(121)
        ax = cluster.single_linkage_tree_.plot(cmap='viridis', colorbar=False)
        ax2 = fig.add_subplot(122)
        ax2 = cluster.minimum_spanning_tree_.plot(edge_cmap='viridis',edge_alpha=0.6, node_size=80,   edge_linewidth=2)
        sns.despine()
        fig.savefig(pic_IObytes,  format='png')
        fig.clear()
        #lpotlyfigured2 = mpl2plotlyGraph(fig)
        pic_IObytes.seek(0)

        graph_part = [ html.Img(src ='data:image/png;base64,{}'.format(base64.b64encode(pic_IObytes.read()).decode()))]#cytodisplay ,cytodisplay1
        #graph_part = [lpotlyfigured2]

        return px.scatter(dfscatter, x="UMAP_1", y="UMAP_2", color = 'hdbscan', hover_data=dfscatter.columns, template='plotly',height=1200, width=1500), df.to_json(), graph_part,cytodisplay2
        #return dcc.Graph(figure= px.scatter(dfscatter, x="UMAP_1", y="UMAP_2", color = 'hdbscan', hover_data=dfscatter.columns, template='plotly',height=1200, width=1500), id = 'umap_plot_selectable'), df.to_json(), graph_part,cytodisplay2
    return px.scatter(x = [0], y = [0]), pd.DataFrame(np.zeros([1,1])).to_json(), html.Div(), html.Div()




@app.callback(Output(component_id= 'selected_points_umap', component_property ='data'),
              Output(component_id= 'umap_selected_stats', component_property ='children'),
              Input(component_id= 'UMAP_view', component_property ='selectedData'),
              State(component_id= 'df_with_umap', component_property ='data'))
def store_selected_umap_points(x, json_df):
    if x and x['points']:
        try:
            region = pd.concat([pd.DataFrame(i) for i in x['points']])
            indices = region.groupby(['curveNumber','pointIndex']).first().customdata.unique()
            dataset = pd.read_json(json_df,convert_dates = False)
            subset = dataset.iloc[indices] #.apply(lambda w: w[0])
            outer_group = dataset.copy().drop(indices, axis = 0)
            subset_describe = subset.describe().fillna(-999).T.reset_index() #include='all'
            subset_describe['ttest_p-value'] = subset_describe['index'].apply(lambda x: ttest_ind(subset[x], outer_group[x], equal_var = False)[1])
            subset_describe['ttest1samp_p-value'] = subset_describe['index'].apply(lambda x: ttest_1samp(subset[x], dataset[x].mean()).pvalue)

            subset_describe = subset_describe[['index', 'count', 'mean', 'std', 'ttest_p-value','ttest1samp_p-value', 'min', 'max', '50%', '25%', '75%']]
        except:
            return pd.DataFrame(np.zeros([1,1])).to_json(), html.Div(str(list(region['customdata'].apply(lambda w: int(w)).unique())))
        return subset.to_json(), dash_table.DataTable( data=subset_describe.to_dict('records'), columns=[{'name': str(i), 'id': str(i),
                                                                                                         'type':'numeric', 'format': Format(precision=5, scheme=Scheme.fixed)
                                                                                                         } for i in subset_describe.columns], style_table={'overflowX': 'auto'},
                                                      style_cell={'minWidth': '180px', 'width': '90px', 'maxWidth': '180px','overflow': 'hidden','textOverflow': 'ellipsis'}, style_as_list_view=True,
                                                     style_data_conditional=[ {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'} ,
                                                                              {'if': {'column_id': 'ttest_p-value',  'filter_query': '{ttest_p-value} <= 0.05'}, 'color': 'red','fontWeight': 'bold'},
                                                                              {'if': {'column_id': 'ttest1samp_p-value',  'filter_query': '{ttest1samp_p-value} <= 0.05'}, 'color': 'red','fontWeight': 'bold'}],
                                                      style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'} )
    else:
        return pd.DataFrame(np.zeros([1,1])).to_json(), html.Div()


@app.callback(Output(component_id= 'prophet_future_dates', component_property ='start_date'),
              Output(component_id= 'prophet_future_dates', component_property ='end_date'),
              Output(component_id= 'prophet_remove_months', component_property ='value'),
              Output(component_id= 'prophet_remove_days_of_the_week', component_property ='value'),
              Output(component_id= 'prophet_remove_hours', component_property ='value'),
              Input(component_id= 'prophet_ds', component_property ='value'),
              Input(component_id= 'prophet_regressors', component_property ='value'),
              State(component_id= 'df', component_property ='data'),
              )
def add_prophet_future(ds_column,regressors, data ):
    try: df = pd.read_json(data)
    except: return datetime.datetime.now().strftime('%b %d %Y'), (datetime.datetime.now() +  + datetime.timedelta(365)).strftime('%b %d %Y'), [],[],[]
    df[ds_column] = df[ds_column].apply(pd.to_datetime)
    if len(regressors)> 0:
        ranged = df[df[ds_column].dt.year == df[ds_column].dt.year.max()].copy()
        start = (ranged[ds_column].min() + datetime.timedelta(365)).strftime('%b %d %Y')
        end = (ranged[ds_column].max() + datetime.timedelta(365)).strftime('%b %d %Y')
    else:
        start = (df[ds_column].max() + datetime.timedelta(1)).strftime('%b %d %Y')
        end = (df[ds_column].max() + datetime.timedelta(366)).strftime('%b %d %Y')
    not_present_months = [x for x in range(1,13) if x not in df[ds_column].dt.month.unique()]
    not_present_weekdays = [x for x in range(7) if x not in df[ds_column].dt.weekday.unique()]
    not_present_hours = [x for x in range(24) if x not in df[ds_column].dt.hour.unique()]

    return start, end, not_present_months, not_present_weekdays, not_present_hours

@app.callback(Output(component_id= 'prophet_plots', component_property ='children'),
              Input(component_id= 'run_prophet', component_property ='n_clicks'),
              State(component_id= 'df', component_property ='data'),
              State(component_id= 'df_with_umap', component_property ='data'), #  ,
              State(component_id= 'prophet_y', component_property ='value'),
              State(component_id= 'prophet_ds', component_property ='value'),
              State(component_id= 'prophet_regressors', component_property ='value'),
              State(component_id= 'prophet_rolling_average', component_property ='value'),
              State(component_id= 'prophet_growth', component_property ='value'),
              State(component_id= 'prophet_cap', component_property ='value'),
              State(component_id= 'prophet_floor', component_property ='value'),
              State(component_id= 'prophet_seasonality', component_property ='value'),
              State(component_id= 'seasonality_mode', component_property ='value'),
              State(component_id= 'season_prior', component_property ='value'),
              State(component_id= 'prophet_n_change_points', component_property ='value'),
              State(component_id= 'changepoint_prior', component_property ='value'),
              State(component_id= 'changepoint_range', component_property ='value'),
              State(component_id= 'prophet_future_dates', component_property ='start_date'),
              State(component_id= 'prophet_future_dates', component_property ='end_date'),
              State(component_id= 'prophet_remove_months', component_property ='value'),
              State(component_id= 'prophet_remove_days_of_the_week', component_property ='value'),
              State(component_id= 'prophet_remove_hours', component_property ='value'))
def run_fbprophet(click,data, data_umap, y_column, ds_column, regressors, rolling_avg, growth, cap, floor, seasonality, season_mode, season_scale, change_points_n, change_points_prior, change_points_range,
                 forecast_range_start,forecast_range_end, removed_months, removed_weekdays, removed_hours):
    if data != None:
        df = pd.read_json(data)# parse_dates=[ds_column]
        df2 = pd.read_json(data_umap) #parse_dates=[ds_column]
        if df2.shape[1] > df.shape[1]:
            df = df2
            ds_column = ds_column.replace(' ', '_')
            y_column = y_column.replace(' ', '_')
            regressors = [x.replace(' ', '_') for x in regressors]
        #if use_ml_pipe == True: preprocessed_data = inpt_children_to_pipe(MLcolumns, MLfuncs, MLclassif)['preprocessing'].fit_transform(df)
        df[ds_column] = df[ds_column].apply(pd.to_datetime)
        df = df.sort_values(ds_column)
        if rolling_avg > 0.1:
            df = df.set_index(ds_column)
            df = df.rolling(window= str(int(rolling_avg *24))+'H').mean().reset_index()
            #resampled_data = fixed_dataset_time.resample('20d').mean().dropna().reset_index()

        df = df[[ds_column, y_column]+ regressors].dropna() ### added dropna()
        df.columns = ['ds', 'y']+ regressors

        season_true_kwards = {x: 25 for x in seasonality}
        season_false_kwards = {x: False for x in ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality'] if x not in seasonality}

        if growth == 'logistic':
            df['cap'] = cap
            df['floor'] = floor

        df = df[~(df['ds'].dt.month.isin(removed_months))]
        df = df[~(df['ds'].dt.weekday.isin(removed_months))]
        df = df.query('ds.dt.hour not in @removed_hours')


        fbmodel = Prophet(growth = growth,seasonality_mode =season_mode, seasonality_prior_scale =season_scale,
                          n_changepoints= change_points_n, changepoint_prior_scale=change_points_prior , changepoint_range = change_points_range,
                          **season_true_kwards, **season_false_kwards) #mcmc_samples=100

        for i in regressors:
            fbmodel.add_regressor(i)

        fbmodel.fit(df)

        if len(regressors) > 0:
            future = df[df.ds.dt.year == df.ds.dt.year.max()].copy()
            future.ds += pd.to_timedelta(365, unit='d')

        elif 'daily_seasonality' not in seasonality:
            future = pd.DataFrame(pd.date_range(pd.to_datetime(forecast_range_start), pd.to_datetime(forecast_range_end),freq='d'), columns = ['ds'])
        else:
            future = pd.DataFrame(pd.date_range(pd.to_datetime(forecast_range_start), pd.to_datetime(forecast_range_end),freq='H'), columns = ['ds'])

        future = future[~(future['ds'].dt.month.isin(removed_months))]
        future = future[~(future['ds'].dt.weekday.isin(removed_months))]
        future = future.query('ds.dt.hour not in @removed_hours')

        if growth == 'logistic':
            future['cap'] = cap
            future['floor'] = floor

        forecast = fbmodel.predict(pd.concat([df, future], axis = 0).reset_index())

        returnable = [dcc.Graph(figure= plot_plotly(fbmodel,forecast, figsize = (1240, 960),  xlabel=ds_column, ylabel=y_column), id = 'prophet_forecast_plot', ),
                      dcc.Graph(figure= plot_components_plotly(fbmodel,forecast, figsize = (1240, 340)), id = 'prophet_forecast_components')]
        if len(regressors) > 0:
            regressor_coefs = regressor_coefficients(fbmodel)
            regressor_coefs
            regressor_coefs['coef_abs'] = regressor_coefs.coef.apply(abs)
            regressor_coefs = regressor_coefs.sort_values('coef_abs', ascending = False)
            #sns.barplot(x = 'regressor', y = 'coef', data =regressor_coefs)
            fig00 = px.bar(regressor_coefs, x="regressor", y="coef", hover_data=regressor_coefs.columns, template='plotly',height=800, width=1240)
            returnable += [dcc.Graph(figure = fig00, id='regressor_impt')] #[mplfig2html(fig1)]

        returnable_tabs = dcc.Tabs([dcc.Tab(children = content, label = name) for name,content in zip(['timeline', 'components', 'regressor coeficients'], returnable)])

        return returnable_tabs

    return html.Div()


@app.callback(Output(component_id= 'dataset_mapping', component_property ='value'),
              Input(component_id= 'df', component_property ='data'),
              Input(component_id= 'df_with_umap', component_property ='data'),
              Input(component_id= 'selected_points_umap', component_property ='data'))
def update_whichdf(a,b,c):
    return 'df'

@app.callback(Output(component_id= 'lat_mapping', component_property ='options'),
              Output(component_id= 'lon_mapping', component_property ='options'),
              Output(component_id= 'lat_mapping', component_property ='value'),
              Output(component_id= 'lon_mapping', component_property ='value'),
              Output(component_id= 'color_mapping', component_property ='options'),
              Output(component_id= 'size_mapping', component_property ='options'),
              Output(component_id= 'time_mapping', component_property ='options'),
              Input(component_id= 'dataset_mapping', component_property ='value'),
              State(component_id= 'df', component_property ='data'),
              State(component_id= 'df_with_umap', component_property ='data'),
              State(component_id= 'selected_points_umap', component_property ='data'))
def update_whichdf(whichdf,data, datau, umap_selelection):
    try:
        if whichdf == 'df':
            df = pd.read_json(data,convert_dates = False)
            ret = [{'label': str(1), 'value': str(1)}]

        elif whichdf == 'df with umap':
            df = pd.read_json(datau,convert_dates = False)
            ret = [{'label': str(0), 'value': str(0)}]

        elif whichdf == 'selected points of umap plot':
            df = pd.read_json(umap_selelection,convert_dates = False)
            ret = [{'label': str(2), 'value': str(2)}]

        latname = [x for x in df.columns if ('latitude' in x.lower()) ] + [x for x in df.columns if ('lat' in x.lower()) ]
        lonname = [x for x in df.columns if ('longitude' in x.lower()) ] + [x for x in df.columns if ('lon' in x.lower()) ]
        latname = latname[0] if len(latname)>=1 else None
        lonname = lonname[0] if len(lonname)>=1 else None


    except: return [],[],None,None,[],[], []
    options = [{'label': str(i), 'value': str(i)} for i in sorted(df.columns)]
    options2 = [{'label': str(i), 'value': str(i)} for i in ['none']+sorted(df.columns) ]
    return options,options,latname, lonname,options2,options2,options2


@app.callback(Output(component_id= 'keplermap', component_property ='children'),
              Input(component_id= 'run_keplermap',component_property= 'n_clicks'),
              State(component_id= 'dataset_mapping', component_property ='value'),
              State(component_id= 'lat_mapping', component_property ='value'),
              State(component_id= 'lon_mapping', component_property ='value'),
              State(component_id= 'color_mapping', component_property ='value'),
              State(component_id= 'size_mapping', component_property ='value'),
              State(component_id= 'time_mapping', component_property ='value'),
              State(component_id= 'df', component_property ='data'),
              State(component_id= 'df_with_umap', component_property ='data'),
              State(component_id= 'selected_points_umap', component_property ='data'))
def Generate_map(n_clicks,whichdf, lat, lon, color,size,date_column ,data, datau, umap_selelection):
    try:
        store_dict = {}
        if whichdf == 'df':
            df = pd.read_json(data,convert_dates = False)

        elif whichdf == 'df with umap':
            df = pd.read_json(datau,convert_dates = False)

        elif whichdf == 'selected points of umap plot':
            df = pd.read_json(umap_selelection,convert_dates = False)

        df = df.dropna(subset = [x for x in [lat, lon, color,size,date_column] if x != 'none'])

        if date_column != 'none':
            df[date_column] = pd.to_datetime(df[date_column])
            store_dict['animation_frame'] = date_column

        if color != 'none':
            store_dict['color'] = color

        if size != 'none':
            store_dict['size'] = 'size'
            try:
                df['size'] = MinMaxScaler(feature_range=(1.5, 15)).fit_transform(df[[size]])
            except:
                df['size'] = make_pipeline(OrdinalEncoder(),MinMaxScaler(feature_range=(1.5, 15))).fit_transform(df[[size]])
    except:
        return html.Div()

    try:
        fig = px.scatter_mapbox(df, lat=lat,  lon=lon, hover_data=df.columns, zoom=10,
                                color_continuous_scale=px.colors.sequential.Turbo ,
                                color_discrete_sequence=px.colors.qualitative.D3,
                                **store_dict)
        fig.update_layout( autosize=False,width=1550,height=1100, mapbox = {'style':'carto-positron'}, )
    except:
        return html.H3('failed creating map')

    return dcc.Graph(figure = fig, id='testtest' )#  height='1343', width='2380'

if __name__ == '__main__':
    app.run_server(debug=True)
