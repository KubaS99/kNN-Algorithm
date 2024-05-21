using System.Globalization;

namespace kNN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var knn = new kNN();
            knn.ReadDataset(@"iris.csv", ',');

            //Testing single random test set
            //var testSet = zad.GetTestSet(0.1);
            //zad.Predict(1, testSet, Metric.Euclidian);
            //zad.TestPrint(testSet);

            //Testing kNN with cross-validation
            knn.CrossValidation(11, 10, Metrics.Euclidian);
            
        }
    }
}