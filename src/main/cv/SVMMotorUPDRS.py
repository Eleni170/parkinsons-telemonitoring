from cv.CrossValidationSVM import CrossValidationSVM

if __name__ == '__main__':
    application = CrossValidationSVM('../../dataset/parkinsons_updrs.data', 6)
    application.cv_svm_motor_updrs()
