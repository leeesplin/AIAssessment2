using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;


namespace MLNet
{
    class Program
    {

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            IDataView dataView =
                mlContext.Data.LoadFromTextFile<InputData>("iris_dataset.csv", hasHeader: true, separatorChar: ',');

            var featureVectorName = "Features";
            var labelColumnName = "Labels";
            var pipeline =
                mlContext.Transforms.Conversion.MapValueToKey(
                        inputColumnName: "Species",
                        outputColumnName: labelColumnName)
                    .Append(mlContext.Transforms.Concatenate(featureVectorName,
                        "SepalLength",
                        "SepalWidth",
                        "PetalLength",
                        "PetalWidth"))
                    .AppendCacheCheckpoint(mlContext)
                    .Append(mlContext.MulticlassClassification.Trainers
                        .SdcaMaximumEntropy(labelColumnName, featureVectorName))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(dataView);

            using (var fileStream = new FileStream("model.zip", FileMode.Create, FileAccess.Write,
                FileShare.Write)) { mlContext.Model.Save(model, dataView.Schema, fileStream); }

            var predictor = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(model);

            //var prediction = predictor.Predict(new InputData()
            //{
            //    SepalLength = 5.3f,
            //    SepalWidth = 3.6f,
            //    PetalLength = 1.2f,
            //    PetalWidth = 0.3f,
            //    Species = "I. setosa"
            //});

            //var prediction = predictor.Predict(new InputData()
            //{
            //    SepalLength = 6.5f,
            //    SepalWidth = 2.4f,
            //    PetalLength = 5.7f,
            //    PetalWidth = 1.9f,
            //    Species = "I. virginica"
            //});

            var prediction = predictor.Predict(new InputData()
            {
                SepalLength = 4.8f,
                SepalWidth = 0f,
                PetalLength = 1.1f,
                PetalWidth = 0.2f,
                Species = "I. versicolor"
            });

            Console.WriteLine($"*** Prediction: {prediction.Species} ***");
            Console.WriteLine($"*** Scores: {string.Join(" ", prediction.Scores)} ***");

            Console.ReadLine();
        }
    }
}
