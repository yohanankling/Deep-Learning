from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


def logistic_regression(train_images, train_labels, test_images, test_labels):
    # Reshape images for logistic regression
    flat_train_images = train_images.reshape((train_images.shape[0], -1))
    flat_test_images = test_images.reshape((test_images.shape[0], -1))

    # Train logistic regression model
    model = LogisticRegression(max_iter=250)
    model.fit(flat_train_images, train_labels.argmax(axis=1))

    # Evaluate logistic regression model
    predictions = model.predict(flat_test_images)
    accuracy = accuracy_score(test_labels.argmax(axis=1), predictions)
    print(f'Logistic Regression Accuracy: {accuracy}')

    # Logistic regression Loss
    loss = log_loss(test_labels, model.predict_proba(flat_test_images))
    print(f'Logistic Regression Loss: {loss}')
