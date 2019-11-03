from app.ApplicationSVM import ApplicationSVM

if __name__ == '__main__':
    application = ApplicationSVM('../dataset/parkinsons_updrs.data', 6)
    application.main_svm_total_updrs()
