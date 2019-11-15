from gs.GridSearchCVSVM import GridSearchCVSVM

if __name__ == '__main__':
    application = GridSearchCVSVM('../../dataset/parkinsons_updrs.data', 6)
    application.gs_svm_total_updrs()
