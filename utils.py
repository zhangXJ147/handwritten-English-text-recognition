import nltk

def CER(label, prediction):
    return nltk.edit_distance(label, prediction) / len(label)

def WER(label, prediction):
    if label == prediction:
        return 0.0
    else:
        return 1.0
    # return nltk.edit_distance(prediction.split(' '), label.split(' ')) / len(label.split(' '))