using System;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using System.IO;

// CMP304: Artificial Intelligence  - Lab 2 Example Code

namespace FeatureExtraction
{
    // The main program class
    class FeatureExtraction
    {
        // file paths
        private const string inputFilePath = "input.jpg";

        private string[] mugImages = Directory.GetFiles("MUGImages", "*.jpg", SearchOption.AllDirectories);

        // The main program entry point
        static void Main(string[] args)
        {
            FeatureExtraction FE = new FeatureExtraction();

            // Set up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape Detector
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                //the header definition of the csv file
                string header = "label,leftEyebrow,rightEyebrow,leftLip,rightLip,lipHeight,LipWidth\n";

                //create the csv file and fill in the first line with the header
                System.IO.File.WriteAllText(@"feature_vectors.csv", header);

                for (int j = 0; j < FE.mugImages.Length; j++)
                { // load input image
                var img = Dlib.LoadImage<RgbPixel>(FE.mugImages[j]);


                    // find all faces in the image
                    var faces = fd.Operator(img);
                    // for each face draw over the facial landmarks
                    foreach (var face in faces)
                    {
                        // find the landmark points for this face
                        var shape = sp.Detect(img, face);

                        // draw the landmark points on the image
                        for (var i = 0; i < shape.Parts; i++)
                        {
                            var point = shape.GetPart((uint)i);
                            var rect = new Rectangle(point);

                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                        }

                        var leftEyebrow1 = distance(shape, 19, 40) / distance(shape, 22, 40);
                        var leftEyebrow2 = distance(shape, 20, 40) / distance(shape, 22, 40);
                        var leftEyebrow3 = distance(shape, 21, 40) / distance(shape, 22, 40);
                        var leftEyebrow4 = distance(shape, 22, 40) / distance(shape, 22, 40);

                        var leftEyebrowFeature = leftEyebrow1 + leftEyebrow2 + leftEyebrow3 + leftEyebrow4;

                        var rightEyebrow1 = distance(shape, 23, 43) / distance(shape, 23, 43);
                        var rightEyebrow2 = distance(shape, 24, 43) / distance(shape, 23, 43);
                        var rightEyebrow3 = distance(shape, 25, 43) / distance(shape, 23, 43);
                        var rightEyebrow4 = distance(shape, 26, 43) / distance(shape, 23, 43);

                        var rightEyebrowFeature = rightEyebrow1 + rightEyebrow2 + rightEyebrow3 + rightEyebrow4;

                        var leftLip1 = distance(shape, 34, 49) / distance(shape, 34, 52);
                        var leftLip4 = distance(shape, 34, 50) / distance(shape, 34, 52);
                        var leftLip3 = distance(shape, 34, 51) / distance(shape, 34, 52);
                        var leftLip2 = distance(shape, 34, 52) / distance(shape, 34, 52);

                        var leftLipFeature = leftLip1 + leftLip2 + leftLip3 + leftLip4;

                        var rightLip1 = distance(shape, 34, 52) / distance(shape, 34, 52);
                        var rightLip2 = distance(shape, 34, 53) / distance(shape, 34, 52);
                        var rightLip3 = distance(shape, 34, 54) / distance(shape, 34, 52);
                        var rightLip4 = distance(shape, 34, 55) / distance(shape, 34, 52);

                        var rightLipFeature = leftLip1 + leftLip2 + leftLip3 + leftLip4;

                        var lipWidthFeature = distance(shape, 49, 55) / distance(shape, 34, 52);

                        var lipHeightFeature = distance(shape, 52, 58) / distance(shape, 34, 52);

                        using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"feature_vectors.csv", true))
                        {
                            file.WriteLine("," + leftEyebrowFeature + "," + rightEyebrowFeature + "," + leftLipFeature + "," + rightLipFeature + "," + lipHeightFeature + "," + lipWidthFeature);
                        }

                    }

                    // export the modified image
                    Dlib.SaveJpeg(img, "output.jpg");
                }
            }

            double distance(FullObjectDetection shape, uint i, uint j)
            {
                return Math.Sqrt(Math.Pow((shape.GetPart(j).X - shape.GetPart(i).X), 2) + Math.Pow((shape.GetPart(j).Y - shape.GetPart(i).Y), 2));
            }

        }
    }
}