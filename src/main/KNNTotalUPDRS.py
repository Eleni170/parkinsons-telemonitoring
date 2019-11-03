from app.ApplicationKNN import ApplicationKNN

if __name__ == '__main__':
    application = ApplicationKNN('../dataset/parkinsons_updrs.data', 6)
    application.main_knn_total_updrs()
