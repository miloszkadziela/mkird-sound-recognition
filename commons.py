import pandas as pd
import numpy as np
import librosa
import librosa.feature

def load(fileName):
    return librosa.load(fileName, sr=22050, duration=5)
    
def convertSound(function, X_, **params):
    def toDataFrame(X):
        return pd.DataFrame(X.values.tolist(), index=X.index)
  
    return pd.concat([toDataFrame(X_.apply(lambda sound: function(sound[0], sr=sound[1], **params).mean(axis=1))),
                      toDataFrame(X_.apply(lambda sound: function(sound[0], sr=sound[1], **params).std(axis=1))),
                      toDataFrame(X_.apply(lambda sound: np.median(function(sound[0], sr=sound[1], **params), axis=1))),
                      toDataFrame(X_.apply(lambda sound: librosa.feature.melspectrogram(sound[0], sr=sound[1]).mean(axis=1))),
                      toDataFrame(X_.apply(lambda sound: librosa.feature.melspectrogram(sound[0], sr=sound[1]).std(axis=1))),
                      toDataFrame(X_.apply(lambda sound: np.median(librosa.feature.melspectrogram(sound[0], sr=sound[1]), axis=1))),
                      toDataFrame(X_.apply(lambda sound: librosa.feature.zero_crossing_rate(sound[0]).mean())),
                      toDataFrame(X_.apply(lambda sound: librosa.feature.zero_crossing_rate(sound[0]).std()))], axis=1)

class SoundRecognizer:
    
    def __init__(self, classifier, scaler, dimensionalityReduction, categories, function, **params):
        self.classifier = classifier
        self.scaler = scaler
        self.dimensionalityReduction = dimensionalityReduction
        self.categories = categories
        self.function = function
        self.params = params
        
    def __convert(self, X):
        return self.dimensionalityReduction.transform(self.scaler.transform(convertSound(self.function, X, **self.params)))
        
    def fileToX(self, file):
        return pd.DataFrame(file, columns=['X'])['X'].apply(lambda file: load(file))
            
    def classifyFile(self, file):
        return self.classify(self.fileToX(file))

    def classify(self, X):
        return [self.categories[i] for i in self.classifier.predict(self.__convert(X))]