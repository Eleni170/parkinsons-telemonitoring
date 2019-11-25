from gs.GridSearchCVKNN import GridSearchCVKNN

if __name__ == '__main__':
    application = GridSearchCVKNN('dataset/parkinsons_updrs.data', 6)
    application.gs_knn_total_updrs()
