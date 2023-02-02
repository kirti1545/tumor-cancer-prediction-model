from django.shortcuts import render
import numpy as np
import pandas as pd
# Create your views here.

import pickle

model = pickle.load(open('breast_cancer_detection.pickle', 'rb'))


def breast_cancer_detection(request):
    if request.method=='POST':

        mean_radius=float(request.POST['mean_radius'])
        mean_texture=float(request.POST['mean_texture'])
        mean_perimeter=float(request.POST['mean_perimeter'])
        mean_area=float(request.POST['mean_area'])
        mean_smoothness=float(request.POST['mean_smoothness'])
        mean_compactness=float(request.POST['mean_compactness'])
        mean_concavity=float(request.POST['mean_concavity'])
        mean_concave_points=float(request.POST['mean_concave_points'])
        mean_symmetry=float(request.POST['mean_symmetry'])
        mean_fractal_dimension=float(request.POST['mean_fractal_dimension'])
        radius_error=float(request.POST['radius_error'])
        texture_error=float(request.POST['texture_error'])
        perimeter_error=float(request.POST['perimeter_error'])
        area_error=float(request.POST['area_error'])
        smoothness_error=float(request.POST['smoothness_error'])
        compactness_error=float(request.POST['compactness_error'])
        concavity_error=float(request.POST['concavity_error'])
        concave_points_error=float(request.POST['concave_points_error'])
        symmetry_error=float(request.POST['symmetry_error'])
        fractal_dimension_error=float(request.POST['fractal_dimension_error'])
        worst_radius=float(request.POST['worst_radius'])
        worst_texture=float(request.POST['worst_texture'])
        worst_perimeter=float(request.POST['worst_perimeter'])
        worst_area=float(request.POST['worst_area'])
        worst_smoothness=float(request.POST['worst_smoothness'])
        worst_compactness=float(request.POST['worst_compactness'])
        worst_concavity=float(request.POST['worst_concavity'])
        worst_concave_points=float(request.POST['worst_concave_points'])
        worst_symmetry=float(request.POST['worst_symmetry'])
        worst_fractal_dimension=float(request.POST['worst_fractal_dimension'])


        # input_features = [float(x) for x in float(request.POST.form.values()]
        input_features=[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            mean_compactness,
            mean_concavity,
            mean_concave_points,
            mean_symmetry,
            mean_fractal_dimension,
            radius_error,
            texture_error,
            perimeter_error,
            area_error,
            smoothness_error,
            compactness_error,
            concavity_error,
            concave_points_error,
            symmetry_error,
            fractal_dimension_error,
            worst_radius,
            worst_texture,
            worst_perimeter,
            worst_area,
            worst_smoothness,
            worst_compactness,
            worst_concavity,
            worst_concave_points,
            worst_symmetry,
            worst_fractal_dimension
        ]
        features_value = [np.array(input_features)]
    
        features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
        df = pd.DataFrame(features_value, columns=features_name)
        output = model.predict(df)
        
        if output == 0:
           res_val = "** Breast Cancer **"
        else:
           res_val = "No Breast Cancer"
        
        context={
            'prediction_text':'Patient has {}'.format(res_val)
        }
        

        return render(request,'home.html',context)
    else:
        return render(request,'home.html')
