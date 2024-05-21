using ArffTools;
using CsvHelper.Configuration;
using CsvHelper;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net.WebSockets;

namespace kNN
{
    //Class to provide metrics for kNN
    public static class Metrics
    {
        public static double Euclidian(Vector v1, Vector v2)
        {
            double sum = 0;
            for (int i = 0; i < v1.features.Count; i++)
            {
                var tmp = Math.Abs(v1.features[i] - v2.features[i]);
                tmp *= tmp;
                sum += tmp;
            }
            return Math.Sqrt(sum);
        }

        public static double Manhattan(Vector v1, Vector v2)
        {
            double sum = 0;
            for (int i = 0; i < v1.features.Count; i++)
            {
                var tmp = Math.Abs(v1.features[i] - v2.features[i]);
                sum += tmp;
            }
            return sum;
        }
    }

    //Class to represent single vector of dataset
    public class Vector
    {
        public int id;
        public List<double> features;
        public string vClass;
        public string predictedClass = "";

        public Vector(int id, List<double> features, string vClass)
        {
            this.id = id;
            this.features = features;
            this.vClass = vClass.Replace("\"","");
        }

        public void Print()
        {
            Console.Write("ID: " + id+" ");
            foreach(var v in features)
                Console.Write(v+" ");
            Console.Write("Class: " + vClass + "\n");
        }
    }

    
    internal class kNN
    {
        List<Vector> vectors;

        //helper method to read dataset from csv file
        public void ReadDataset(string fileName, char delimiter, bool hasHeader = true)
        {
            vectors = new List<Vector>();
            var reader = new StreamReader(fileName);
            if (hasHeader)
            {
                reader.ReadLine();
            }
            int counter = 0;

            
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(delimiter);
                var features = new List<double>();
                for(int i=0;i<values.Length-1;i++)
                {
                    if (values[i][0] == '.')
                        values[i] = "0" + values[i];
                    features.Add(double.Parse(values[i], CultureInfo.InvariantCulture.NumberFormat));
                }
                vectors.Add(new Vector(counter, features, values[values.Length-1]));
                counter++;
            }
            
        }

        //method to get random test set from dataset
        public List<Vector> GetTestSet(double ratio)
        {
            List<Vector> result = new List<Vector>();
            int num = (int)(ratio * vectors.Count);
            Random r = new Random();
            for(int i=0;i<num;i++)
            {
                int index = r.Next(vectors.Count);
                result.Add(vectors[index]);
                vectors.Remove(vectors[index]);
            }
            return result;
        }

       
        private Vector GetVector(int id)
        {
            foreach(var v in vectors)
            {
                if (v.id == id)
                    return v;
            }
            return null;
        }

        //method to perform knn algorithm 
        public void Predict(int k,List<Vector> testSet, Func<Vector,Vector, double> metric)
        {
            foreach(var t in testSet)
            {
                Dictionary<int, double> distances = new Dictionary<int, double>();
                foreach (var v in vectors)
                {
                    distances[v.id] = metric(t, v);
                }
                Dictionary<string, int> scores = new Dictionary<string, int>();
                int counter = 0;

                //sorting calculated distances ascending and adding k nearest neighbours to scores dictionary that contains class names of training vectors
                foreach (KeyValuePair<int, double> entry in distances.OrderBy(key => key.Value))
                {
                    string tmpClass = GetVector(entry.Key).vClass;
                    if (scores.ContainsKey(tmpClass))
                        scores[tmpClass]++;
                    else
                        scores[tmpClass] = 1;
                    counter++;
                    if (!(counter < k))
                    {
                        counter = 0;
                        break;
                    }
                        
                }

                //determining predicted class
                string maxClass="";
                int max = int.MinValue;
                foreach(var i in scores.Keys)
                {
                    if (scores[i]>max)
                    {
                        max= scores[i];
                        maxClass = i;
                    }
                }
                t.predictedClass = maxClass;

            }
        }

        //method to perform k-folds cross-validation
        public void CrossValidation(int k,int folds, Func<Vector, Vector, double> metric)
        {
            double result=0;
            List<List<Vector>> dataSets = new List<List<Vector>>();
            for(int i=0;i<folds;i++)
            {
                dataSets.Add( new List<Vector>());
            }

            //determining number of vectors in each data subset
            int num = vectors.Count / folds;

            //creating k subsets
            Random r = new Random();
            for(int i=0;i<folds;i++)
            {
                for (int j = 0; j < num; j++)
                {
                    if (vectors.Count != 0)
                    {
                        var index = r.Next(vectors.Count);
                        dataSets[i].Add(vectors[index]);
                        vectors.Remove(vectors[index]);
                    }
                    else
                        Console.WriteLine("i: " + i + " j: " + j);
                }
            }
            vectors.Clear();

            //merging non test subsets into training set
            for(int i=0;i<folds;i++)
            {
                var testSet = dataSets[i];
                for(int j=0;j<folds;j++)
                {
                    if(j!=i)
                    {
                        foreach(Vector v in dataSets[j])
                        {
                            vectors.Add(v);
                        }
                    }
                }

                //testing test set
                Predict(k, testSet, metric);
                var tmp = Test(testSet);
                Console.WriteLine("Fold "+i+": "+Math.Round(tmp,2)+"%");
                result += Test(testSet);
                vectors.Clear();       
            }
            result/=folds;
            Console.WriteLine("Accuracy: " + Math.Round(result, 2) + "%");

        }

        //method to test results of kNN algorithm and print them
        public void TestPrint(List<Vector> testSet)
        {
            int sum=0;
            foreach(var v in testSet)
            {
                Console.WriteLine("Vector id: {0,4}| Class: {1,15}| Predicted class: {2,15}",v.id,v.vClass,v.predictedClass);
                if (v.vClass == v.predictedClass)
                    sum++;
            }
            double result = (double)sum/(double)testSet.Count*100;
            Console.WriteLine("Accuracy: " + Math.Round(result, 2) + "%");
        }

        //method to test results of kNN algorithm in cross-validation
        public double Test(List<Vector> testSet)
        {
            int sum = 0;
            foreach (var v in testSet)
            {
                if (v.vClass == v.predictedClass)
                    sum++;
            }
            double result = (double)sum / (double)testSet.Count * 100;
            return result;
        }

    }
}
