from cv.CrossValidationKNN import CrossValidationKNN

if __name__ == '__main__':
    application = CrossValidationKNN('dataset/parkinsons_updrs.data', 6)
    application.cv_knn_motor_updrs()
