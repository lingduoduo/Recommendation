import tensorflow as tf

# import scann


def predict_train_scann_model(viewer, path):
    model = tf.saved_model.load(f"model/{path}")
    scores, broadcasters = model(
        {
            "viewer": tf.constant([viewer]),
        }
    )
    scores = scores.numpy()[0]
    broadcasters = broadcasters.numpy()[0]
    preds = {}
    for i in range(len(scores)):
        preds[str(broadcasters[i])] = str(scores[i])
    print(",".join(preds.keys()))
    return preds
